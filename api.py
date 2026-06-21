import gc
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from load_pipeline import (
    OUTPUT_DIR,
    generate_ms_17b_video,
    generate_video,
    load_ms_17b_text_to_video_pipeline,
    load_text_to_video_pipeline,
)


app = FastAPI(title="ModelScope Text-to-Video API")

DEMO_MODEL = "demo"
MS_17B_MODEL = "ms_17b"

_demo_pipe = None
_ms_17b_pipe = None
_active_model = None
_generation_lock = Lock()


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)


class GenerateMs17bRequest(GenerateRequest):
    inf_steps: Optional[int] = Field(default=None, ge=1)
    frames: Optional[int] = Field(default=None, ge=1)


class GenerateResponse(BaseModel):
    text: str
    video_path: str
    host_path: str


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
