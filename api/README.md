# Purple AI inference API

This API serves the **custom model trained by** `training/train_from_scratch.py`. It does not download or rename Qwen or another pretrained model.

## Important: a trained checkpoint is required

The repository currently contains training code, not trained weights. Train the model first. By default the API looks for:

`checkpoints/purple_ai_latest.pt`

If the checkpoint is stored elsewhere, set `PURPLE_CHECKPOINT_PATH` to its path. Keep the checkpoint and training vocabulary/config from the same run.

## Install and run (Windows CMD)

From the repository root:

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\purple_ai_latest.pt
set PURPLE_API_KEY=replace-this-with-a-long-random-secret
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation and `http://127.0.0.1:8000/health` to check whether the checkpoint loaded.

To test generation, send a POST request to `/generate` with JSON such as:

```json
{"prompt":"Pytanie: Czym jest komputer?\nOdpowiedź:","max_new_chars":200,"temperature":0.8}
```

If `PURPLE_API_KEY` is set, send header `Authorization: Bearer YOUR_KEY`. Never publish the key or commit it to GitHub.

## Endpoints

- `GET /` — API info
- `GET /health` — server/model status
- `POST /generate` — generates a continuation of the supplied text

## Limitations

This is a tiny character-level next-character model, not yet a capable instruction-following chatbot. It continues text rather than reliably answering questions. Unknown characters in a prompt are ignored. Do not expose this development server publicly without proper authentication, HTTPS, rate limits, and hosting configuration. The API is not hosted by GitHub; this command runs it on the computer where you execute it.
