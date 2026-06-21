import pathlib
import shutil
import subprocess
import uuid
from datetime import datetime, timezone

from huggingface_hub import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline


MODEL_REPO = "damo-vilab/modelscope-damo-text-to-video-synthesis"
MODEL_DIR = pathlib.Path("weights")
OUTPUT_DIR = pathlib.Path("outputs")
ORIG_OUTPUT_SUBDIR = "orig"
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


def load_text_to_video_pipeline():
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


def generate_video(pipe, text, output_dir=OUTPUT_DIR):
    output_dir = pathlib.Path(output_dir)
    orig_output_dir = output_dir / ORIG_OUTPUT_SUBDIR
    prompt = {
        "text": text,
    }

    result = pipe(prompt)
    generated_path = pathlib.Path(result[OutputKeys.OUTPUT_VIDEO])

    output_dir.mkdir(parents=True, exist_ok=True)
    orig_output_dir.mkdir(parents=True, exist_ok=True)
    extension = generated_path.suffix or ".mp4"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_id = uuid.uuid4().hex[:8]
    raw_output_path = orig_output_dir / f"{timestamp}-{output_id}{extension}"
    output_path = output_dir / f"{timestamp}-{output_id}.mp4"

    if generated_path.resolve() != raw_output_path.resolve():
        shutil.copy2(generated_path, raw_output_path)

    try:
        convert_video(raw_output_path, output_path)
    finally:
        if generated_path.resolve() != raw_output_path.resolve():
            delete_file(generated_path)

    return output_path


if __name__ == "__main__":
    pipe = load_text_to_video_pipeline()
    print(f"Loaded text-to-video pipeline from {MODEL_DIR.resolve()}")

    video_path = generate_video(pipe, DEFAULT_PROMPT)
    print("Saved video at:", video_path)
