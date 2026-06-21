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

Converted videos are saved into `./outputs`; the original generated files are kept in `./outputs/orig`. File names use `MMDD-hhmmss-prompt.mp4`; whitespace in the prompt becomes `_`, and only the first 64 prompt characters are used. Files from the demo model use a `demo_` prefix.

## Extract frames

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/enchance" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"filename":"0621-134200-Spiderman_is_surfing.mp4","steps":1}'
```

Frames are written to `./outputs/frames` as `00001.png`, `00002.png`, and so on. Existing image files in `./outputs/upscaled` are deleted, the loaded text-to-video pipeline is released from VRAM, then all extracted frames are uploaded and queued in ComfyUI on the host at `http://host.docker.internal:8188` using `workflows/image_upgrade.api.json`. ComfyUI still executes the queued frame prompts sequentially; the API collects outputs in frame order. If one fails or times out, the API stops, best-effort removes pending queued frame prompts, and returns an error. When all frames finish, ffmpeg combines `./outputs/upscaled/*.png` into `./outputs/upscaled/<input filename>`, then `workflows/frame_interpolation.api.json` is queued in ComfyUI and the returned video is copied into `./outputs/interpolated` with the same filename.

## Interpolate a video

The input file must be an `.mp4` inside `./outputs/upscaled`.

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/interpolate" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"filename":"0621-134200-Spiderman_is_surfing.mp4"}'
```

```bash
curl -X POST "http://127.0.0.1:8080/interpolate" \
  -H "Content-Type: application/json" \
  -d '{"filename":"0621-134200-Spiderman_is_surfing.mp4"}'
```

The video is sent to ComfyUI on the host using `workflows/frame_interpolation.api.json`. The returned video is saved to `./outputs/interpolated` with the same filename.
