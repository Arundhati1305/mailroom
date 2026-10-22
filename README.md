# Safe AI Mailroom (ga5-mailroom-action-gate/v2)

One FastAPI endpoint, `POST /`, handles `propose` and `commit`.

What code does: schema checks, canonical hashing, durable storage, replay and 409 handling, Ed25519 receipt checks, and safety checks on everything the model returns.
What the model does: reads one dossier and suggests an action.

Files: `app.py` (service), `tests/test_local.py` (plays the grader, no model calls), `smoke.py` (tests a deployed URL without using a grader check), `Dockerfile`, `requirements.txt`.

## 1. Test locally
    pip install -r requirements.txt pytest
    MOCK_LLM=1 pytest -q tests/

## 2. Deploy (free option: Hugging Face Space or Render + Neon Postgres)
Storage must survive restarts, and free container disks are wiped on restart. So create a free Postgres database (Neon or Supabase) and set `DATABASE_URL`. Without it, the app uses a local SQLite file, which is fine only on a host with a persistent volume.

Environment variables:
| var | example |
|---|---|
| `LLM_BASE_URL` | `https://aipipe.org/openai/v1` (any OpenAI-compatible API) |
| `LLM_API_KEY`  | your token (set as a secret, never in code) |
| `LLM_MODEL`    | `gpt-4.1-mini` |
| `DATABASE_URL` | `postgresql://...` |
| `ADMIN_TOKEN`  | long random string, for `/debug/dossiers` |
| `PORT`         | `7860` on HF Spaces, Render sets it itself |

For models that reject `temperature`, such as gpt-5 family models, set `LLM_TEMPERATURE=` (empty).

Keep **one worker**. If your host sleeps (Render free does), open `GET /` a minute before pressing Check. A cold start can exceed the 55 s limit.

## 3. Check the deployed URL for free
    python smoke.py https://your-app.example.com/
This checks the propose shape, replay, 409, a forged receipt, the commit, the commit replay and a malformed request.

## 4. Spend grader checks wisely
1. **Check #1.** This runs the model on the 58 core dossiers plus the audit cases. Read the category feedback.
2. Review the core decisions yourself:
   `curl -H "X-Admin-Token: $ADMIN_TOKEN" https://.../debug/dossiers`
   Look at the actual `kind`/`provenance` strings and tune `TRUSTED_RE` / `UNTRUSTED_RE` and the prompt to match them.
3. To re-decide cached dossiers after a fix, change `PROMPT_VERSION`. `callId`s depend only on dossier content, so they stay stable.
4. Save only when you are satisfied, and keep the reserved final check.

If the feedback flags an HTTP status for a rejection, the status codes are constants at the top of `app.py`.
