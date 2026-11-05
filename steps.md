Here's the order to do it in. The first three steps happen on your laptop, then everything moves to Colab.

## Step 1: Commit and push the code

Notebook 06 downloads your code from GitHub (`develop` branch), so it only sees changes you've pushed.

1. Review the changes with `git status` and `git diff`, then commit and push to `develop` yourself.
2. Is the repo private? Then create a GitHub token: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained → read access to the `EduHinglish` repo. Keep it for Step 5.

## Step 2: Set up Hugging Face

1. Log in at https://huggingface.co, open https://huggingface.co/google/gemma-3-4b-it and click **Acknowledge license**. Skip this if you did it for v3.
2. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Write** → copy it.
3. Decide your two repo names. For example, if your HF username is `ashvatth`:
   - Gemma model: `ashvatth/eduhinglish-gemma-3-4b-it-lora`
   - Intent model: `ashvatth/eduhinglish-muril-intent`

## Step 3: Upload the intent model (laptop, once)

The demo notebook needs this model, and its weights aren't in git.

```bash
cd ~/AshWorks/EduHinglish
source eduhinglish_env/bin/activate
pip install -U huggingface_hub
huggingface-cli login        # paste the WRITE token (on newer versions: hf auth login)
python scripts/push_to_hub.py intent models/muril_intent_v3 ashvatth/eduhinglish-muril-intent
```

It uploads about 950 MB. When it finishes it prints `[OK] https://huggingface.co/...`.

## Step 4: Upload the data to Google Drive

In Google Drive, open `MyDrive/EduHinglish/data/` and upload these five files from your local `data/` folder:

`bio_9_10.json`, `chem_9_10.json`, `phys_9_10.json`, `maths_9_10.json`, `ss_9_10.json`

Also check that `MyDrive/EduHinglish/models/gemma_eduhinglish_v3/adapter/` still exists. Training uses it as its starting point. If it's gone, training starts from scratch, which also works.

## Step 5: Add Colab secrets

Open https://colab.research.google.com and upload `training/colab_notebooks/05_gemma_finetune_v4_resumable.ipynb`. Click the 🔑 **Secrets** icon in the left sidebar and add:

| Name | Value |
|---|---|
| `HF_TOKEN` | your HF **write** token |
| `GITHUB_TOKEN` | only if the repo is private (from Step 1) |

Turn on **Notebook access** for each one. Secrets belong to your Google account, so the demo notebook will see them too.

## Step 6: Train the model (notebook 05)

1. Runtime → Change runtime type → **T4 GPU**.
2. Run **only the first code cell** (the pip install). When it finishes: Runtime → **Restart session**.
3. In the **Settings** cell, change `HF_REPO_ID = "ashvatth/eduhinglish-gemma-3-4b-it-lora"`. Leave everything else.
4. Runtime → **Run all**. Allow Drive access when asked.
5. Check the output of the "Training state" cell:
   - `No checkpoint yet — starting a new run.` means a new run. On later runs it shows `Resuming from ... step X / Y`.
6. Training prints the loss every 10 steps and saves to Drive every 50 steps (about every 6 minutes). Total is about 860 steps.

**If Colab disconnects** (it will, on the free tier):
1. Reconnect to a T4.
2. Run the pip cell again, then Restart session.
3. Runtime → **Run all**. It continues from the last saved step, losing at most a few minutes of work.

**When it finishes**, the last cells:
- save the model to `MyDrive/EduHinglish/models/gemma_eduhinglish_v4/adapter/`
- push it to your HF repo (check that it appears at huggingface.co/ashvatth/...)
- print one sample answer per subject next to the reference answer. Read these to judge quality.

After that, running the notebook again skips training.

## Step 7: Run the demo (notebook 06), any time you need it

1. Upload `training/colab_notebooks/06_inference_demo.ipynb` to Colab and pick a **T4 GPU** runtime.
2. Run the pip cell, then Restart session.
3. In **Settings**, set:
   ```python
   HF_ADAPTER_REPO = "ashvatth/eduhinglish-gemma-3-4b-it-lora"
   HF_INTENT_REPO  = "ashvatth/eduhinglish-muril-intent"
   ```
4. **Run all**. How long it takes:
   - **First time:** about 10 minutes, mostly building the knowledge base, which is then saved to Drive (`knowledge_base_v4/`).
   - **After that:** about 3–5 minutes.
5. The sample cell answers one question per subject.
6. The last cell prints a link like `https://abcd1234.gradio.live`. Open it on any phone or laptop for the demo. Pick the **Subject** in the dropdown for better answers.

The link works only while the notebook is running. For a demo, start the notebook about 10 minutes beforehand.

## Step 7b: Run the model as an API and connect the backend

Notebook 06 is the Gradio demo page. Notebook 07 runs the same model as a plain HTTP API, which `backend/` calls.

```
Frontend ──► backend/ (your laptop or a server) ──► Colab notebook 07 (T4 GPU, public URL)
```

**Once: add secrets in Colab** (🔑 sidebar):

| Name | Value |
|---|---|
| `MODEL_API_KEY` | a long random string, e.g. from `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `NGROK_AUTHTOKEN` | optional, for a URL that doesn't change (see below) |

**Optional: a fixed URL**
Without this, the URL changes every Colab session and you have to update `backend/.env` each time.
1. Sign up at https://ngrok.com (free).
2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken into the `NGROK_AUTHTOKEN` secret.
3. Create your free static domain at https://dashboard.ngrok.com/domains. Put it in `NGROK_DOMAIN` in the notebook's Settings cell, e.g. `eduhinglish-demo.ngrok-free.app`.

**Each session: start the model server**
1. Upload `training/colab_notebooks/07_colab_model_server.ipynb` to Colab and pick a **T4 GPU** runtime.
2. Run the pip cell, then Restart session.
3. In Settings, set `HF_ADAPTER_REPO` and `HF_INTENT_REPO`.
4. **Run all**.
5. Cell 6 prints `MODEL_SERVER_URL = https://...`. It also saves the URL to `MyDrive/EduHinglish/model_server_url.txt`.
6. Cell 7 asks one question through the public URL. You should see a Hinglish answer.
7. Leave cell 8 (the heartbeat) running. Stopping it or closing the runtime takes the API down.

**Start the backend** (laptop):

```bash
cd ~/AshWorks/EduHinglish
source eduhinglish_env/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
- `MODEL_SERVER_URL` = the URL from step 5
- `MODEL_API_KEY` = the same value as the Colab secret

Then start it:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

**Check it works**
- Open http://localhost:8000. This is the chat page, served by the backend itself (no Node, no build step). The dot at the bottom left turns green when the Colab model is reachable.
- Open http://localhost:8000/api/health. `model_server.status` should be `"ok"`, with `lora_loaded: true`.
- Open http://localhost:8000/docs → `POST /api/ask` → **Try it out** with:
  ```json
  {"query": "Ohm ka law kya hai?", "subject": "physics", "class_num": "10"}
  ```
- Or from a terminal:
  ```bash
  curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" \
       -d '{"query": "Ohm ka law kya hai?", "subject": "physics", "class_num": "10"}'
  ```

**What the frontend (and later the Flutter app) calls** (backend at `http://<backend-host>:8000`):

| Endpoint | Body | Returns |
|---|---|---|
| `GET /api/health` | — | backend status + model server status |
| `POST /api/chat` | `message`, optional `chat_id` (omit for a new chat), `subject`, `class_num`, `language` | `chat`, `user_message`, `assistant_message` (`content`, `intent`, `sources[]`, `timings`, `feedback`). Nothing is saved if the model fails, so just retry |
| `GET /api/chats` | — | past chats, newest first |
| `GET /api/chats/{id}` | — | one chat with all its `messages` |
| `PATCH /api/chats/{id}` | any of `title`, `subject`, `class_num`, `language` | updated chat |
| `DELETE /api/chats/{id}` | — | 204 |
| `POST /api/messages/{id}/feedback` | `value`: 1 (👍), -1 (👎), 0 (clear) | updated message |
| `POST /api/ask` | `query`, optional `subject` (biology/chemistry/physics/mathematics/social_science), `class_num` ("9"/"10"), `language` (hinglish/english) | `answer`, `intent`, `sources[]` (chapter, subject, class_num, score), `timings` |
| `POST /api/preprocess` | `sentence` | M1 trace (script, normalisation, LID…) |

An answer takes about 10–40 s on a T4, so show a loading state in the frontend.

Chats and feedback are stored in SQLite at `backend/data/eduhinglish.db` (git-ignored; change with `DB_PATH`).

If Colab is off, `/api/chat` and `/api/ask` return **503** with "Model server unreachable … Is the Colab notebook running?".

## Step 8: Publish the dataset on Kaggle (laptop, anytime)

1. kaggle.com → profile → **Settings** → API → **Create New Token**. This downloads `kaggle.json`.
2. Install it:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   pip install kaggle
   python scripts/publish_kaggle_dataset.py <kaggle-username>/eduhinglish-ncert-hinglish-qa --with-text
   ```
3. It's private at first. Add `--public` on that first upload, or change visibility later on Kaggle.
4. When the data changes, run it again with `-m "what changed"` to upload a new version.

## Step 9: Retrain the intent model (later, optional)

The intent model has never seen physics or maths questions. After the main training is done, follow steps 1–3 in [PIPELINE_RUNBOOK_v3.md](workspace/PIPELINE_RUNBOOK_v3.md). Then run Step 3 above again to upload the new model.

---

**Quick checklist:** push code → HF licence and write token → upload intent model → upload 5 JSONs to Drive → Colab secrets → notebook 05 (Run all until done) → notebook 06 (demo page) or notebook 07 + backend (API) → Kaggle.

If any cell fails, paste the error here.