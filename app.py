"""HTTP boundary for the single-source CV website and Google AI Studio chat proxy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from cv_domain import (
    MAX_HISTORY_TURNS,
    MAX_MESSAGE_CHARS,
    SlidingWindowRateLimiter,
    build_system_prompt,
    format_cv_markdown,
    load_cv,
    normalize_history,
)
from llm_client import (
    DEFAULT_MODEL,
    LLMConfigError,
    LLMError,
    generate_reply,
    stream_reply,
)

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(env_path: Path) -> None:
    """Populate missing environment variables from local .env without overriding host env."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


_load_dotenv(BASE_DIR / ".env")

CV_DATA = load_cv(BASE_DIR / "cv.json")
CV_MARKDOWN = format_cv_markdown(CV_DATA)
SYSTEM_PROMPT = build_system_prompt(CV_MARKDOWN, CV_DATA["identity"]["name"])

app = Flask(__name__)
rate_limiter = SlidingWindowRateLimiter()


def get_client_ip() -> str:
    """Return visitor IP, reading the leftmost X-Forwarded-For hop behind Render's proxy."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def resolve_llm_settings() -> tuple[str, str]:
    """Read Google AI Studio API key and Gemini model from server environment."""
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_STUDIO_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        raise LLMConfigError("LLM_API_KEY is not configured on the server.")
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL
    return api_key, model


def _validate_chat_request() -> tuple[str, list[dict[str, str]]] | tuple[Response, int]:
    if not rate_limiter.allow(get_client_ip()):
        return jsonify({"error": "Too many requests. Please wait a minute and try again."}), 429

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body."}), 400

    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Message must be a non-empty string."}), 400

    cleaned = message.strip()
    if len(cleaned) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message must be {MAX_MESSAGE_CHARS} characters or fewer."}), 400

    try:
        history = normalize_history(body.get("history"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return cleaned, history


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
def home():
    """Serve the CV website rendered directly from cv.json."""
    return render_template(
        "index.html",
        cv=CV_DATA,
        max_message_chars=MAX_MESSAGE_CHARS,
        max_history_turns=MAX_HISTORY_TURNS,
    )


@app.get("/healthz")
def healthz():
    """Health check endpoint for Render."""
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL
    return jsonify({"status": "ok", "model": model}), 200


@app.get("/cv.md")
def cv_markdown():
    """Serve deterministic Markdown projection of the canonical CV."""
    return Response(CV_MARKDOWN + "\n", status=200, mimetype="text/markdown; charset=utf-8")


@app.get("/cv.json")
def cv_json():
    """Serve canonical structured CV JSON."""
    return jsonify(CV_DATA), 200


@app.post("/api/chat")
def chat():
    """Handle single-shot 'Chat with My CV' requests behind the server trust boundary."""
    validated = _validate_chat_request()
    if isinstance(validated[0], Response):
        return validated
    cleaned, history = validated

    try:
        api_key, model = resolve_llm_settings()
        reply = generate_reply(api_key, model, SYSTEM_PROMPT, history, cleaned)
        return jsonify({"reply": reply}), 200
    except LLMError as exc:
        return jsonify({"error": str(exc)}), exc.http_status


@app.post("/api/chat/stream")
def chat_stream():
    """Stream 'Chat with My CV' reply tokens over Server-Sent Events."""
    validated = _validate_chat_request()
    if isinstance(validated[0], Response):
        return validated
    cleaned, history = validated

    try:
        api_key, model = resolve_llm_settings()
    except LLMError as exc:
        return jsonify({"error": str(exc)}), exc.http_status

    def event_stream():
        chunks: list[str] = []
        try:
            for delta in stream_reply(api_key, model, SYSTEM_PROMPT, history, cleaned):
                chunks.append(delta)
                yield _sse_event("delta", {"text": delta})
            final_reply = "".join(chunks).strip() or "Sorry, I could not generate a response."
            yield _sse_event("done", {"reply": final_reply})
        except LLMError as exc:
            yield _sse_event("error", {"error": str(exc)})

    return Response(
        stream_with_context(event_stream()),
        status=200,
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
