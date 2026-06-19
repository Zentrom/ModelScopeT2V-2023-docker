FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MODELSCOPE_CACHE=/cache/modelscope \
    TORCH_HOME=/cache/torch

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install modelscope==1.4.2 \
    && python -m pip install open_clip_torch \
    && python -m pip install pytorch-lightning \
    && python -m pip install huggingface_hub

COPY . /workspace

CMD ["python"]
