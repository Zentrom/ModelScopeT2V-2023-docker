import gc
import json
import os
from pathlib import Path
from time import monotonic, sleep
from threading import Lock
from typing import Optional
from urllib import error, parse, request
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from load_pipeline import (
    OUTPUT_DIR,
    clear_upscaled_outputs,
    extract_video_frames,
    generate_ms_17b_video,
    generate_video,
    load_ms_17b_text_to_video_pipeline,
    load_text_to_video_pipeline,
)


app = FastAPI(title="ModelScope Text-to-Video API")

DEMO_MODEL = "demo"
MS_17B_MODEL = "ms_17b"
COMFYUI_BASE_URL = os.getenv("COMFYUI_URL", "http://host.docker.internal:8188").rstrip("/")
COMFYUI_TIMEOUT_SECONDS = 10
COMFYUI_POLL_INTERVAL_SECONDS = 5
COMFYUI_POLL_TIMEOUT_SECONDS = 300
IMAGE_UPGRADE_WORKFLOW_PATH = Path("workflows") / "image_upgrade.api.json"
IMAGE_UPGRADE_KSAMPLER_NODE_ID = "3"
IMAGE_UPGRADE_LOAD_IMAGE_NODE_ID = "78"
IMAGE_UPGRADE_SAVE_IMAGE_NODE_ID = "60"

_demo_pipe = None
_ms_17b_pipe = None
_active_model = None
_request_lock = Lock()
_generation_lock = Lock()


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)


class GenerateMs17bRequest(GenerateRequest):
    inf_steps: Optional[int] = Field(default=None, ge=1)
    frames: Optional[int] = Field(default=None, ge=1)


class EnchanceRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    steps: Optional[int] = Field(default=None, ge=1)


class GenerateResponse(BaseModel):
    text: str
    video_path: str
    host_path: str


class EnchanceResponse(BaseModel):
    filename: str
    steps: int
    video_path: str
    frames_path: str
    host_frames_path: str
    first_frame_path: str
    host_first_frame_path: str
    frame_pattern: str
    comfyui_url: str
    comfyui_uploaded_image: str
    comfyui_prompt_id: str
    comfyui_output_filename: str
    comfyui_output_subfolder: str
    comfyui_output_type: str
    upscaled_image_path: str
    host_upscaled_image_path: str


@app.middleware("http")
async def reject_concurrent_requests(request, call_next):
    if not _request_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={
                "detail": "API is already processing another request.",
            },
        )

    try:
        return await call_next(request)
    finally:
        _request_lock.release()


def _find_output_file(filename):
    clean_filename = Path(filename).name
    if not clean_filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    output_path = OUTPUT_DIR / clean_filename
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="File not found in outputs.")

    return output_path


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _comfyui_request(path, data=None, content_type="application/json", method=None):
    headers = {}
    body = None
    if data is not None:
        body = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
        headers["Content-Type"] = content_type

    url = f"{COMFYUI_BASE_URL}{path}"
    req = request.Request(
        url,
        data=body,
        headers=headers,
        method=method or ("POST" if data is not None else "GET"),
    )

    try:
        with request.urlopen(req, timeout=COMFYUI_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI rejected the request at {COMFYUI_BASE_URL}: {detail}",
        ) from exc
    except error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to ComfyUI at {COMFYUI_BASE_URL}: {exc}",
        ) from exc

    if not response_body:
        return {}

    return json.loads(response_body)


def _comfyui_get_bytes(path):
    url = f"{COMFYUI_BASE_URL}{path}"
    req = request.Request(url, method="GET")

    try:
        with request.urlopen(req, timeout=COMFYUI_TIMEOUT_SECONDS) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI rejected the request at {COMFYUI_BASE_URL}: {detail}",
        ) from exc
    except error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to ComfyUI at {COMFYUI_BASE_URL}: {exc}",
        ) from exc


def _encode_multipart_form_data(fields, files):
    boundary = f"----modelscope-t2v-{uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for name, path in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _upload_image_to_comfyui(image_path):
    body, content_type = _encode_multipart_form_data(
        fields={
            "type": "input",
            "overwrite": "true",
        },
        files={
            "image": image_path,
        },
    )
    response = _comfyui_request("/upload/image", data=body, content_type=content_type)
    image_name = response.get("name") or image_path.name
    subfolder = response.get("subfolder")
    if subfolder:
        return f"{subfolder}/{image_name}"

    return image_name


def _configure_image_upgrade_workflow(workflow, uploaded_image, steps=None):
    try:
        workflow[IMAGE_UPGRADE_LOAD_IMAGE_NODE_ID]["inputs"]["image"] = uploaded_image
        ksampler_inputs = workflow[IMAGE_UPGRADE_KSAMPLER_NODE_ID]["inputs"]
    except KeyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Image upgrade workflow is missing a required node.",
        ) from exc

    if steps is not None:
        ksampler_inputs["steps"] = steps

    return ksampler_inputs["steps"]


def _submit_image_upgrade_to_comfyui(image_path, steps=None):
    if not IMAGE_UPGRADE_WORKFLOW_PATH.is_file():
        raise HTTPException(status_code=500, detail="Image upgrade workflow not found.")

    uploaded_image = _upload_image_to_comfyui(image_path)
    workflow = _read_json(IMAGE_UPGRADE_WORKFLOW_PATH)
    configured_steps = _configure_image_upgrade_workflow(
        workflow,
        uploaded_image,
        steps=steps,
    )

    response = _comfyui_request(
        "/prompt",
        data={
            "prompt": workflow,
            "client_id": uuid4().hex,
        },
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI did not return a prompt_id: {response}",
        )

    return uploaded_image, prompt_id, configured_steps


def _wait_for_comfyui_prompt(prompt_id):
    deadline = monotonic() + COMFYUI_POLL_TIMEOUT_SECONDS
    history_path = f"/history/{parse.quote(prompt_id)}"

    while monotonic() < deadline:
        history_response = _comfyui_request(history_path)
        history = history_response.get(prompt_id)
        if history:
            status = history.get("status") or {}
            status_str = status.get("status_str")
            completed = status.get("completed")
            if status_str == "error":
                raise HTTPException(
                    status_code=500,
                    detail=f"ComfyUI prompt failed: {status}",
                )

            if completed or history.get("outputs"):
                return history

        sleep(COMFYUI_POLL_INTERVAL_SECONDS)

    raise HTTPException(
        status_code=504,
        detail=f"Timed out waiting for ComfyUI prompt {prompt_id}.",
    )


def _find_comfyui_output_image(history):
    outputs = history.get("outputs") or {}
    save_output = outputs.get(IMAGE_UPGRADE_SAVE_IMAGE_NODE_ID) or {}
    images = save_output.get("images") or []
    if images:
        return images[0]

    for output in outputs.values():
        images = output.get("images") or []
        if images:
            return images[0]

    raise HTTPException(
        status_code=500,
        detail="ComfyUI completed but did not return an output image.",
    )


def _download_comfyui_image(image_info, output_path):
    params = parse.urlencode(
        {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
    )
    image_bytes = _comfyui_get_bytes(f"/view?{params}")
    output_path.write_bytes(image_bytes)


def _release_pipeline(pipe):
    if pipe is None:
        return

    maybe_free_model_hooks = getattr(pipe, "maybe_free_model_hooks", None)
    if callable(maybe_free_model_hooks):
        maybe_free_model_hooks()


def _clear_memory():
    gc.collect()

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _unload_inactive_pipeline(model_name):
    global _active_model, _demo_pipe, _ms_17b_pipe

    unloaded = False
    if model_name != DEMO_MODEL and _demo_pipe is not None:
        _release_pipeline(_demo_pipe)
        _demo_pipe = None
        unloaded = True

    if model_name != MS_17B_MODEL and _ms_17b_pipe is not None:
        _release_pipeline(_ms_17b_pipe)
        _ms_17b_pipe = None
        unloaded = True

    if unloaded:
        _active_model = None
        _clear_memory()


def _get_demo_pipeline():
    global _active_model, _demo_pipe

    _unload_inactive_pipeline(DEMO_MODEL)
    if _demo_pipe is None:
        _demo_pipe = load_text_to_video_pipeline()

    _active_model = DEMO_MODEL
    return _demo_pipe


def _get_ms_17b_pipeline():
    global _active_model, _ms_17b_pipe

    _unload_inactive_pipeline(MS_17B_MODEL)
    if _ms_17b_pipe is None:
        _ms_17b_pipe = load_ms_17b_text_to_video_pipeline()

    _active_model = MS_17B_MODEL
    return _ms_17b_pipe


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _active_model is not None,
        "demo_model_loaded": _demo_pipe is not None,
        "ms_17b_model_loaded": _ms_17b_pipe is not None,
        "active_model": _active_model,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")

    try:
        with _generation_lock:
            pipe = _get_demo_pipeline()
            video_path = generate_video(pipe, text, output_dir=OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    host_path = Path("outputs") / video_path.name
    return GenerateResponse(
        text=text,
        video_path=video_path.resolve().as_posix(),
        host_path=host_path.as_posix(),
    )


@app.post("/enchance", response_model=EnchanceResponse)
def enchance(request: EnchanceRequest):
    video_path = _find_output_file(request.filename.strip())
    upscaled_dir = clear_upscaled_outputs(output_dir=OUTPUT_DIR)

    try:
        frames_dir, frame_pattern = extract_video_frames(video_path, output_dir=OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    first_frame_path = frames_dir / "00001.png"
    if not first_frame_path.is_file():
        raise HTTPException(status_code=500, detail="No frames were extracted.")

    uploaded_image, prompt_id, configured_steps = _submit_image_upgrade_to_comfyui(
        first_frame_path,
        steps=request.steps,
    )
    history = _wait_for_comfyui_prompt(prompt_id)
    output_image = _find_comfyui_output_image(history)
    upscaled_image_path = upscaled_dir / first_frame_path.name
    _download_comfyui_image(output_image, upscaled_image_path)

    return EnchanceResponse(
        filename=video_path.name,
        steps=configured_steps,
        video_path=video_path.resolve().as_posix(),
        frames_path=frames_dir.resolve().as_posix(),
        host_frames_path=(Path("outputs") / "frames").as_posix(),
        first_frame_path=first_frame_path.resolve().as_posix(),
        host_first_frame_path=(Path("outputs") / "frames" / first_frame_path.name).as_posix(),
        frame_pattern=frame_pattern.resolve().as_posix(),
        comfyui_url=COMFYUI_BASE_URL,
        comfyui_uploaded_image=uploaded_image,
        comfyui_prompt_id=prompt_id,
        comfyui_output_filename=output_image.get("filename", ""),
        comfyui_output_subfolder=output_image.get("subfolder", ""),
        comfyui_output_type=output_image.get("type", "output"),
        upscaled_image_path=upscaled_image_path.resolve().as_posix(),
        host_upscaled_image_path=(
            Path("outputs") / "upscaled" / upscaled_image_path.name
        ).as_posix(),
    )


@app.post("/generate/ms-1.7b", response_model=GenerateResponse)
def generate_ms_17b(request: GenerateMs17bRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")

    try:
        with _generation_lock:
            pipe = _get_ms_17b_pipeline()
            video_path = generate_ms_17b_video(
                pipe,
                text,
                output_dir=OUTPUT_DIR,
                inf_steps=request.inf_steps,
                frames=request.frames,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    host_path = Path("outputs") / video_path.name
    return GenerateResponse(
        text=text,
        video_path=video_path.resolve().as_posix(),
        host_path=host_path.as_posix(),
    )
