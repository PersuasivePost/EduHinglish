# Running EduHinglish locally

The model runs on a Colab T4 GPU. The backend and the chat page run on your laptop.

```
Browser ──► http://localhost:8000 ──► backend/ (FastAPI + chat page + SQLite) ──HTTPS──► Colab notebook 07 (model)
```

| Part | Where | How you start it |
|---|---|---|
| Model server (M1 intent → M2 ChromaDB → M3 Gemma + LoRA) | Colab, T4 GPU | `training/colab_notebooks/07_colab_model_server.ipynb` |
| Backend API and chat page | laptop, port 8000 | `uvicorn backend.app.main:app --port 8000` |
| Chat history and 👍/👎 | laptop, `backend/data/eduhinglish.db` | created automatically |

---

## 1. One-time setup

### Laptop

```bash
cd ~/AshWorks/EduHinglish
source eduhinglish_env/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Generate an API key. The laptop and Colab must use the same value:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Hugging Face

1. Open https://huggingface.co/google/gemma-3-4b-it and click **Acknowledge license**.
2. Make a **read** token at https://huggingface.co/settings/tokens. If your repos are private, it must come from the account that owns them.

### Colab secrets

Open the 🔑 sidebar and add these. Turn on **Notebook access** for each one.

| Name | Value |
|---|---|
| `HF_TOKEN` | your HF read token |
| `MODEL_API_KEY` | the key you generated above |
| `NGROK_AUTHTOKEN` | optional. Leave it out to use Cloudflare, the default. Add it for a URL that stays the same (see section 4). |
| `GITHUB_TOKEN` | only if the GitHub repo is private |

### Notebook 07 Settings cell

```python
HF_ADAPTER_REPO = "<your-hf-user>/<your-gemma-lora-repo>"
HF_INTENT_REPO  = None    # or "<your-hf-user>/eduhinglish-muril-intent" once uploaded
NGROK_DOMAIN    = ""      # or "your-name.ngrok-free.app" (see section 4)
```

The notebook clones the `develop` branch from GitHub. Push your code changes before you run it.

---

## 2. Every time: start everything

### Step 1: Colab (about 5 minutes, or about 10 the first time)

1. Upload `training/colab_notebooks/07_colab_model_server.ipynb` to Colab, or reopen it.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the **first cell** (pip install), then **Runtime → Restart session**.
4. **Runtime → Run all**. Allow Google Drive access when asked.
5. Wait for the cell that prints:
   ```
   MODEL_SERVER_URL = https://xxxx.trycloudflare.com     (Cloudflare: new URL every session)
   MODEL_SERVER_URL = https://your-name.ngrok-free.app   (ngrok: always the same)
   ```
6. The test cell after it should print a Hinglish answer.
7. **Leave the last cell (heartbeat) running.** If you stop it or close the runtime, the model goes offline.

### Step 2: Laptop

Paste the URL into `backend/.env`. With Cloudflare you do this every session. With ngrok you do it once.

```bash
MODEL_SERVER_URL=https://xxxx.trycloudflare.com
MODEL_API_KEY=<same key as the Colab secret>
```

Start the backend. Run it from the repo root, not from inside `backend/`:

```bash
cd ~/AshWorks/EduHinglish
source eduhinglish_env/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

You can also start it with uv instead of the venv:

```bash
uv run --with-requirements backend/requirements.txt uvicorn backend.app.main:app --port 8000
```

### Step 3: Open it

| URL | What it is |
|---|---|
| http://localhost:8000 | the chat page |
| http://localhost:8000/docs | API docs, where you can try each route |
| http://localhost:8000/api/health | backend and model status |

The dot at the bottom left of the chat page shows the model status. **Green** means the model is online. **Red** means the backend can't reach Colab. Hover over the dot to see the reason.

Each answer takes about 10–40 seconds on a T4.

---

## 3. Checking from the terminal

```bash
# Is the model reachable? Look for "status": "ok" and "lora_loaded": true
curl -s localhost:8000/api/health | python -m json.tool

# Ask a question (this creates a new chat)
curl -s -X POST localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "Ohm ka law kya hai?", "subject": "physics", "class_num": "10"}' | python -m json.tool

# Ask without saving to history
curl -s -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"query": "Photosynthesis kaise hota hai?", "subject": "biology"}' | python -m json.tool

# List saved chats
curl -s localhost:8000/api/chats | python -m json.tool

# Call the Colab model server directly, without the backend
curl -s https://xxxx.trycloudflare.com/health
```

---

## 4. The tunnel: Cloudflare (default) or ngrok

Colab has no public address of its own. Notebook 07 opens a tunnel so the backend can reach the model server. It picks the tunnel by itself:

- **No `NGROK_AUTHTOKEN` secret: Cloudflare quick tunnel.** This is the default. You don't need an account or any setup. The URL looks like `https://xxxx.trycloudflare.com` and **changes every Colab session**, so you paste the new one into `backend/.env` and restart uvicorn each time.
- **`NGROK_AUTHTOKEN` secret and `NGROK_DOMAIN` set: ngrok static domain.** Free. The URL **never changes**, so you set `.env` once.

| | Cloudflare quick tunnel | ngrok static domain |
|---|---|---|
| Account | not needed | free account |
| Setup | none | one-time setup, steps below |
| URL | new every session | always the same |
| Each session | copy the URL into `.env`, restart backend | nothing |

**Use Cloudflare to start.** Switch to ngrok once copying the URL every session gets annoying.

Both are public URLs. `MODEL_API_KEY` is what stops strangers from using your GPU, so keep it set.

> **Why not Tailscale?** It keeps the model server private, but it needs much more setup:
> - `tailscaled` has to be installed in Colab every session, in userspace-networking mode
> - you need an auth key
> - your laptop also has to join the tailnet
> - old Colab nodes have to be cleaned up, or the hostname changes
>
> The API key already protects the server, so none of that is worth it for a demo.

### Switching to ngrok (one time)

1. Sign up at https://ngrok.com. It's free.
2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken into the Colab secret `NGROK_AUTHTOKEN`.
3. Create a free static domain at https://dashboard.ngrok.com/domains.
4. Put it in notebook 07: `NGROK_DOMAIN = "your-name.ngrok-free.app"`.
5. Set `MODEL_SERVER_URL=https://your-name.ngrok-free.app` in `backend/.env` once.

---

## 5. Stopping and resetting

| What | How |
|---|---|
| Stop the backend | `Ctrl+C` in the uvicorn terminal |
| Stop the model | stop the heartbeat cell, then **Runtime → Disconnect and delete runtime** |
| Delete all chat history | stop the backend, then `rm backend/data/eduhinglish.db`. A new empty one is created on the next start. |
| Keep the history somewhere else | set `DB_PATH=/some/path.db` in `backend/.env` |

The free Colab tier disconnects after about 90 minutes of inactivity, or about 12 hours in total. When that happens, repeat section 2, Step 1. If you don't use ngrok, also update the URL in `.env`.

---

## 6. Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| Red dot, or "Model server unreachable … Is the Colab notebook running?" | Colab stopped, or the URL in `.env` is old | Rerun notebook 07, paste the new URL, restart uvicorn |
| "MODEL_SERVER_URL is not set" | `backend/.env` is missing or empty | `cp backend/.env.example backend/.env` and fill it in |
| "Model server rejected the API key" | The keys don't match | `MODEL_API_KEY` in `.env` must equal the Colab secret |
| "Model server returned a non-JSON page" | The tunnel is up but Colab died, or ngrok is showing its warning page | Rerun notebook 07 |
| "Model server timed out" | The model is busy with other questions (it answers one at a time) | Wait and press **Retry** |
| `lora_loaded: false` in health | `HF_ADAPTER_REPO` is wrong, or the token can't see the private repo | Check the repo id and `HF_TOKEN`, then rerun the notebook |
| Colab: Gemma download fails with 401/403 | Gemma license not accepted, or `HF_TOKEN` is missing | Accept the license on the Gemma page and check the secret |
| Colab: CUDA out of memory | Not on a T4, or an old session is still using the GPU | Pick T4, then **Disconnect and delete runtime** and rerun |
| `ModuleNotFoundError: backend` | uvicorn was started from the wrong folder | Run it from the repo root |
| Port 8000 already in use | Another uvicorn is still running | `lsof -i :8000`, then kill it, or use `--port 8001` |
