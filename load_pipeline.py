import pathlib
import shutil
import subprocess
import uuid
from datetime import datetime, timezone

from huggingface_hub import snapshot_download


MODEL_REPO = "damo-vilab/modelscope-damo-text-to-video-synthesis"
MS_17B_MODEL_REPO = "ali-vilab/text-to-video-ms-1.7b"
MODEL_DIR = pathlib.Path("weights")
OUTPUT_DIR = pathlib.Path("outputs")
ORIG_OUTPUT_SUBDIR = "orig"
DEMO_OUTPUT_PREFIX = "demo_"
DEFAULT_MS_17B_INFERENCE_STEPS = 25
DEFAULT_PROMPT = "A robot walking through a futuristic city at night, cinematic lighting"


def convert_video(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path.as_posix(),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "main",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        output_path.as_posix(),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")


def delete_file(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def build_output_paths(output_dir, extension=".mp4", filename_prefix=""):
    output_dir = pathlib.Path(output_dir)
    orig_output_dir = output_dir / ORIG_OUTPUT_SUBDIR

    output_dir.mkdir(parents=True, exist_ok=True)
    orig_output_dir.mkdir(parents=True, exist_ok=True)

    extension = extension or ".mp4"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_id = uuid.uuid4().hex[:8]
    file_stem = f"{filename_prefix}{timestamp}-{output_id}"

    return (
        orig_output_dir / f"{file_stem}{extension}",
        output_dir / f"{file_stem}.mp4",
    )


def load_text_to_video_pipeline():
    from modelscope.pipelines import pipeline

    # Download model weights locally so ModelScope can load them from disk.
    snapshot_download(
        MODEL_REPO,
        repo_type="model",
        local_dir=MODEL_DIR,
    )

    return pipeline(
        task="text-to-video-synthesis",
        model=MODEL_DIR.as_posix(),
    )


def load_ms_17b_text_to_video_pipeline():
    import torch
    from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

    load_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if torch.cuda.is_available():
        load_kwargs["variant"] = "fp16"

    try:
        pipe = DiffusionPipeline.from_pretrained(MS_17B_MODEL_REPO, **load_kwargs)
    except OSError:
        if "variant" not in load_kwargs:
            raise

        load_kwargs.pop("variant")
        pipe = DiffusionPipeline.from_pretrained(MS_17B_MODEL_REPO, **load_kwargs)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    return pipe


def generate_video(pipe, text, output_dir=OUTPUT_DIR):
    from modelscope.outputs import OutputKeys

    prompt = {
        "text": text,
    }

    result = pipe(prompt)
    generated_path = pathlib.Path(result[OutputKeys.OUTPUT_VIDEO])
    extension = generated_path.suffix or ".mp4"
    raw_output_path, output_path = build_output_paths(
        output_dir,
        extension=extension,
        filename_prefix=DEMO_OUTPUT_PREFIX,
    )

    if generated_path.resolve() != raw_output_path.resolve():
        shutil.copy2(generated_path, raw_output_path)

    try:
        convert_video(raw_output_path, output_path)
    finally:
        if generated_path.resolve() != raw_output_path.resolve():
            delete_file(generated_path)

    return output_path


def generate_ms_17b_video(
    pipe,
    text,
    output_dir=OUTPUT_DIR,
    inf_steps=None,
    frames=None,
):
    from diffusers.utils import export_to_video

    raw_output_path, output_path = build_output_paths(output_dir)

    generation_kwargs = {
        "num_inference_steps": (
            inf_steps if inf_steps is not None else DEFAULT_MS_17B_INFERENCE_STEPS
        ),
    }
    if frames is not None:
        generation_kwargs["num_frames"] = frames

    result = pipe(text, **generation_kwargs)
    video_frames = result.frames
    if getattr(video_frames, "ndim", None) == 5:
        video_frames = video_frames[0]
    elif len(video_frames) > 0 and (
        isinstance(video_frames[0], (list, tuple))
        or getattr(video_frames[0], "ndim", None) == 4
    ):
        video_frames = video_frames[0]

    export_to_video(video_frames, output_video_path=raw_output_path.as_posix())
    convert_video(raw_output_path, output_path)

    return output_path


if __name__ == "__main__":
    pipe = load_text_to_video_pipeline()
    print(f"Loaded text-to-video pipeline from {MODEL_DIR.resolve()}")

    video_path = generate_video(pipe, DEFAULT_PROMPT)
    print("Saved video at:", video_path)
