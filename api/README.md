# Purple AI inference API

This API serves the **custom model trained by** `training/train_from_scratch.py`. It does not download or rename Qwen or another pretrained model.

## Important: a trained checkpoint is required

The repository currently contains training code, not trained weights. By default the API looks for:

`checkpoints/purple_ai_latest.pt`

If the checkpoint is stored elsewhere, set `PURPLE_CHECKPOINT_PATH` to its path. Alternatively, the API can download `purple_ai_latest.pt` or `purpleai_latest.pt` from the Hugging Face model repository set in `HF_MODEL_REPO_ID` (default: `Kamiljdds/purpleaiV1.0`). The file must be a checkpoint produced by this project's training script, including its model weights, vocabulary, and configuration. A model card or source code alone is not enough.

For a private Hugging Face model repository, configure `HF_TOKEN` as a server secret. For a public repository, a token is usually not required.

## Install and run (Windows CMD)

From the repository root:

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\\purple_ai_latest.pt
set PURPLE_API_KEY=replace-this-with-a-long-random-secret
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Diagnose startup

Open `http://127.0.0.1:8000/health`. The response includes `model_loaded`, whether the configured local checkpoint path exists, the Hugging Face repository ID, and a `model_error` message when the model could not be loaded. This helps distinguish a missing checkpoint from a server or authentication issue. Do not share secrets from server environment variables.

If the checkpoint is missing, generation will return HTTP 503 with the setup reason. This is intentional: the API must not pretend that the model is ready when no trained weights are available.

## Test generation

Send a POST request to `/generate` with JSON such as:

```json
{
  "prompt": "Pytanie: Czym jest komputer?\\nOdpowiedź:",
  "max_new_chars": 200,
  "temperature": 0.8
}
```

If `PURPLE_API_KEY` is configured, include the header `Authorization: Bearer YOUR_KEY`. Never publish the key or commit it to GitHub.

## Endpoints

- `GET /` — API information
- `GET /health` — model and setup diagnostics
- `POST /generate` — generates a continuation of the supplied text

## Limitations

This is a tiny character-level next-character model, not yet a capable instruction-following chatbot. It continues text rather than reliably answering questions. Unknown characters in a prompt are ignored. Do not expose this development server publicly without proper authentication, HTTPS, rate limits, and hosting configuration. The API is not hosted by GitHub; this command runs it on the computer where you execute it.
