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

Converted videos are saved into `./outputs`; the original generated files are kept in `./outputs/orig`.
