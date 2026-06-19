# ModelScopeT2V-2023-docker

Docker install of old ModelScopeT2V.

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

Generated videos are copied into `./outputs`.
