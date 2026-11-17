# CV & Google AI Studio Chat Website (MIS 251)

Single-source personal **CV / Resume** web application with an embedded **"Chat with My CV"** assistant powered by **Google AI Studio (Gemini)** and deployed on **Render**.

## Architecture & Trust Boundaries

```text
Visitor / Recruiter Browser (Untrusted)
   │  1. GET /               -> HTML CV rendered from cv.json (dark/light + print PDF)
   │  2. GET /cv.md          -> Deterministic Markdown projection of cv.json
   │  3. POST /api/chat      -> JSON reply (NO API key in browser)
   │  4. POST /api/chat/stream -> SSE streaming reply (NO API key in browser)
   ▼
Render PaaS Web Service (app.py + cv_domain.py + llm_client.py)
   │  • Validates cv.json at startup and generates system prompt from the same source
   │  • Enforces input bounds (<=500 chars, <=6 turns) & sliding-window IP rate limit (15/min, LRU <=1000 IPs)
   │  • Calls Google AI Studio via official google-genai SDK (15s timeout, bounded thinking)
   ▼
Google AI Studio (Gemini API: gemini-3.5-flash-lite)
```

## Files

- `cv.json` — Canonical structured CV data (single source of truth for both HTML UI and AI context).
- `cv_domain.py` — Boundary CV schema parser, Markdown/prompt formatter, history validator, and sliding-window rate limiter.
- `llm_client.py` — Google AI Studio (`google-genai`) client adapter mapping errors to HTTP `429` / `500` / `502` / `504`.
- `app.py` — Flask server endpoints (`/`, `/healthz`, `/cv.md`, `/cv.json`, `/api/chat`, `/api/chat/stream`).
- `templates/index.html` — Responsive CV layout with theme toggle, print-to-PDF styling, `sessionStorage` chat state, and XSS-safe `textContent` SSE rendering.
- `render.yaml` — Render web service blueprint (`gunicorn --workers 1 --threads 4 --timeout 60 app:app`).

## Running Locally

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
# Set LLM_API_KEY (or GOOGLE_API_STUDIO_KEY) in .env
.venv/bin/python app.py
```

Open `http://127.0.0.1:5000`.
