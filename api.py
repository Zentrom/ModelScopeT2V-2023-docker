from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from load_pipeline import OUTPUT_DIR, generate_video, load_text_to_video_pipeline


app = FastAPI(title="ModelScope Text-to-Video API")

_pipe = None
_generation_lock = Lock()


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)


class GenerateResponse(BaseModel):
    text: str
    video_path: str
    host_path: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _pipe is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    global _pipe

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")

    try:
        with _generation_lock:
            if _pipe is None:
                _pipe = load_text_to_video_pipeline()

            video_path = generate_video(_pipe, text, output_dir=OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    host_path = Path("outputs") / video_path.name
    return GenerateResponse(
        text=text,
        video_path=video_path.resolve().as_posix(),
        host_path=host_path.as_posix(),
    )
