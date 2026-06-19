import pathlib

from huggingface_hub import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline


MODEL_REPO = "damo-vilab/modelscope-damo-text-to-video-synthesis"
MODEL_DIR = pathlib.Path("weights")
DEFAULT_PROMPT = "A robot walking through a futuristic city at night, cinematic lighting"


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


def generate_video(pipe, text):
    prompt = {
        "text": text,
    }

    result = pipe(prompt)
    return result[OutputKeys.OUTPUT_VIDEO]


if __name__ == "__main__":
    pipe = load_text_to_video_pipeline()
    print(f"Loaded text-to-video pipeline from {MODEL_DIR.resolve()}")

    video_path = generate_video(pipe, DEFAULT_PROMPT)
    print("Saved video at:", video_path)
