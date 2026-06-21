import gc
import json
import os
from pathlib import Path
from time import monotonic, sleep
from threading import Lock
from typing import List, Optional
from urllib import error, parse, request
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from load_pipeline import (
    OUTPUT_DIR,
    clear_upscaled_outputs,
    create_upscaled_video,
    ensure_interpolated_output_dir,
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
FRAME_INTERPOLATION_WORKFLOW_PATH = Path("workflows") / "frame_interpolation.api.json"
FRAME_INTERPOLATION_LOAD_VIDEO_NODE_ID = "4"
FRAME_INTERPOLATION_SAVE_VIDEO_NODE_ID = "7"

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


class EnchanceFrameResponse(BaseModel):
    frame: str
    frame_path: str
    host_frame_path: str
    comfyui_uploaded_image: str
    comfyui_prompt_id: str
    comfyui_output_filename: str
    comfyui_output_subfolder: str
    comfyui_output_type: str
    upscaled_image_path: str
    host_upscaled_image_path: str


class EnchanceResponse(BaseModel):
    filename: str
    steps: int
    video_path: str
    frames_path: str
    host_frames_path: str
    frame_pattern: str
    comfyui_url: str
    frame_count: int
    upscaled_frame_count: int
    upscaled_video_path: str
    host_upscaled_video_path: str
    interpolated_video_path: str
    host_interpolated_video_path: str
    comfyui_interpolation_uploaded_video: str
    comfyui_interpolation_prompt_id: str
    comfyui_interpolation_output_filename: str
    comfyui_interpolation_output_subfolder: str
    comfyui_interpolation_output_type: str
    upscaled_frames: List[EnchanceFrameResponse]


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


def _guess_content_type(path):
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".mp4":
        return "video/mp4"

    return "application/octet-stream"


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
        body.extend(f"Content-Type: {_guess_content_type(path)}\r\n\r\n".encode("utf-8"))
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _upload_file_to_comfyui(file_path):
    body, content_type = _encode_multipart_form_data(
        fields={
            "type": "input",
            "overwrite": "true",
        },
        files={
            "image": file_path,
        },
    )
    response = _comfyui_request("/upload/image", data=body, content_type=content_type)
    file_name = response.get("name") or file_path.name
    subfolder = response.get("subfolder")
    if subfolder:
        return f"{subfolder}/{file_name}"

    return file_name


def _upload_image_to_comfyui(image_path):
    return _upload_file_to_comfyui(image_path)


def _upload_video_to_comfyui(video_path):
    return _upload_file_to_comfyui(video_path)


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


def _configure_frame_interpolation_workflow(workflow, uploaded_video):
    try:
        workflow[FRAME_INTERPOLATION_LOAD_VIDEO_NODE_ID]["inputs"]["file"] = uploaded_video
    except KeyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Frame interpolation workflow is missing the LoadVideo node.",
        ) from exc


def _submit_frame_interpolation_to_comfyui(video_path):
    if not FRAME_INTERPOLATION_WORKFLOW_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="Frame interpolation workflow not found.",
        )

    uploaded_video = _upload_video_to_comfyui(video_path)
    workflow = _read_json(FRAME_INTERPOLATION_WORKFLOW_PATH)
    _configure_frame_interpolation_workflow(workflow, uploaded_video)

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

    return uploaded_video, prompt_id


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


def _find_comfyui_output_media(history, node_id, media_keys):
    outputs = history.get("outputs") or {}
    save_output = outputs.get(node_id) or {}
    for media_key in media_keys:
        media = save_output.get(media_key) or []
        if media:
            return media[0]

    for output in outputs.values():
        for media_key in media_keys:
            media = output.get(media_key) or []
            if media:
                return media[0]

    raise HTTPException(
        status_code=500,
        detail="ComfyUI completed but did not return the expected output media.",
    )


def _find_comfyui_output_image(history):
    return _find_comfyui_output_media(
        history,
        IMAGE_UPGRADE_SAVE_IMAGE_NODE_ID,
        ("images",),
    )


def _find_comfyui_output_video(history):
    return _find_comfyui_output_media(
        history,
        FRAME_INTERPOLATION_SAVE_VIDEO_NODE_ID,
        ("videos", "gifs", "images"),
    )


def _download_comfyui_media(media_info, output_path):
    params = parse.urlencode(
        {
            "filename": media_info.get("filename", ""),
            "subfolder": media_info.get("subfolder", ""),
            "type": media_info.get("type", "output"),
        }
    )
    media_bytes = _comfyui_get_bytes(f"/view?{params}")
    output_path.write_bytes(media_bytes)


def _download_comfyui_image(image_info, output_path):
    _download_comfyui_media(image_info, output_path)


def _interpolate_video(video_path, interpolated_dir):
    uploaded_video, prompt_id = _submit_frame_interpolation_to_comfyui(video_path)
    history = _wait_for_comfyui_prompt(prompt_id)
    output_video = _find_comfyui_output_video(history)
    interpolated_video_path = interpolated_dir / video_path.name
    _download_comfyui_media(output_video, interpolated_video_path)

    if not interpolated_video_path.is_file():
        raise HTTPException(
            status_code=500,
            detail="ComfyUI interpolated video was not saved.",
        )

    return uploaded_video, prompt_id, output_video, interpolated_video_path


def _enhance_frame(frame_path, upscaled_dir, steps=None):
    uploaded_image, prompt_id, configured_steps = _submit_image_upgrade_to_comfyui(
        frame_path,
        steps=steps,
    )
    history = _wait_for_comfyui_prompt(prompt_id)
    output_image = _find_comfyui_output_image(history)
    upscaled_image_path = upscaled_dir / frame_path.name
    _download_comfyui_image(output_image, upscaled_image_path)

    if not upscaled_image_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI output was not saved for frame {frame_path.name}.",
        )

    return configured_steps, EnchanceFrameResponse(
        frame=frame_path.name,
        frame_path=frame_path.resolve().as_posix(),
        host_frame_path=(Path("outputs") / "frames" / frame_path.name).as_posix(),
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
    interpolated_dir = ensure_interpolated_output_dir(output_dir=OUTPUT_DIR)

    try:
        frames_dir, frame_pattern = extract_video_frames(video_path, output_dir=OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise HTTPException(status_code=500, detail="No frames were extracted.")

    configured_steps = None
    upscaled_frames = []
    for frame_path in frame_paths:
        configured_steps, frame_result = _enhance_frame(
            frame_path,
            upscaled_dir,
            steps=request.steps,
        )
        upscaled_frames.append(frame_result)

    try:
        upscaled_video_path = create_upscaled_video(upscaled_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    (
        uploaded_interpolation_video,
        interpolation_prompt_id,
        interpolation_output_video,
        interpolated_video_path,
    ) = _interpolate_video(upscaled_video_path, interpolated_dir)

    return EnchanceResponse(
        filename=video_path.name,
        steps=configured_steps,
        video_path=video_path.resolve().as_posix(),
        frames_path=frames_dir.resolve().as_posix(),
        host_frames_path=(Path("outputs") / "frames").as_posix(),
        frame_pattern=frame_pattern.resolve().as_posix(),
        comfyui_url=COMFYUI_BASE_URL,
        frame_count=len(frame_paths),
        upscaled_frame_count=len(upscaled_frames),
        upscaled_video_path=upscaled_video_path.resolve().as_posix(),
        host_upscaled_video_path=(
            Path("outputs") / "upscaled" / upscaled_video_path.name
        ).as_posix(),
        interpolated_video_path=interpolated_video_path.resolve().as_posix(),
        host_interpolated_video_path=(
            Path("outputs") / "interpolated" / interpolated_video_path.name
        ).as_posix(),
        comfyui_interpolation_uploaded_video=uploaded_interpolation_video,
        comfyui_interpolation_prompt_id=interpolation_prompt_id,
        comfyui_interpolation_output_filename=interpolation_output_video.get(
            "filename",
            "",
        ),
        comfyui_interpolation_output_subfolder=interpolation_output_video.get(
            "subfolder",
            "",
        ),
        comfyui_interpolation_output_type=interpolation_output_video.get(
            "type",
            "output",
        ),
        upscaled_frames=upscaled_frames,
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
