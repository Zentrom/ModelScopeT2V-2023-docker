FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MODELSCOPE_CACHE=/cache/modelscope \
    TORCH_HOME=/cache/torch \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip wheel \
    && python -m pip install setuptools==69.5.1 \
    && python -m pip install modelscope==1.4.2 \
    && python -m pip install pyarrow==20.0.0 \
    && python -m pip install opencv-python-headless \
    && python -m pip install open_clip_torch==2.24.0 \
    && python -m pip install pytorch-lightning \
    && python -m pip install huggingface_hub \
    && python -m pip install diffusers==0.33.1 transformers accelerate imageio imageio-ffmpeg \
    && python -m pip install fastapi==0.95.2 "uvicorn[standard]==0.22.0"

COPY . /workspace

RUN mkdir -p /workspace/outputs/orig

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
