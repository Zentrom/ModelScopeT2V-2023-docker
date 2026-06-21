# ModelScopeT2V-2023-docker

Docker install of old ModelScopeT2V: https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis

You need ~16GB VRAM to run.

Commercial use not allowed by model creators.

## Run

```bash
docker compose build
docker compose up
```

The API listens on `http://127.0.0.1:8080`.

## Generate a video

Demo model:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/generate" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"A robot walking through a futuristic city at night, cinematic lighting"}'
```

```bash
curl -X POST "http://127.0.0.1:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"A robot walking through a futuristic city at night, cinematic lighting"}'
```

MS 1.7B model:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/generate/ms-1.7b" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"Spiderman is surfing","inf_steps":25,"frames":32}'
```

```bash
curl -X POST "http://127.0.0.1:8080/generate/ms-1.7b" \
  -H "Content-Type: application/json" \
  -d '{"text":"Spiderman is surfing","inf_steps":25,"frames":32}'
```

Converted videos are saved into `./outputs`; the original generated files are kept in `./outputs/orig`. File names use `MMDD-hhmm-prompt.mp4`; whitespace in the prompt becomes `_`, and only the first 64 prompt characters are used. Files from the demo model use a `demo_` prefix.

## Extract frames

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/enchance" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"filename":"0621-1342-Spiderman_is_surfing.mp4","steps":1}'
```

Frames are written to `./outputs/frames` as `00001.png`, `00002.png`, and so on. Existing image files in `./outputs/upscaled` are deleted, then each extracted frame is sent to ComfyUI on the host at `http://host.docker.internal:8188` using `workflows/image_upgrade.api.json`. Frames are processed one by one; if one fails or times out, the API stops and returns an error.
