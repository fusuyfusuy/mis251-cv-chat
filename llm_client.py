"""Google AI Studio (Gemini) SDK adapter behind typed boundary errors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NoReturn

from google import genai
from google.genai import errors, types
import httpx

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_OUTPUT_TOKENS = 1024
REQUEST_TIMEOUT_MS = 15_000


class LLMError(RuntimeError):
    """Base error raised by the LLM boundary."""

    http_status: int = 502


class LLMConfigError(LLMError):
    """Raised when server LLM configuration is missing or invalid."""

    http_status = 500


class LLMRateLimitError(LLMError):
    """Raised when upstream Gemini rejects a request with HTTP 429."""

    http_status = 429


class LLMTimeoutError(LLMError):
    """Raised when upstream Gemini exceeds REQUEST_TIMEOUT_MS."""

    http_status = 504


class LLMUpstreamError(LLMError):
    """Raised when upstream Gemini fails or cannot be reached."""

    http_status = 502


def _thinking_config_for(model: str) -> types.ThinkingConfig | None:
    """Keep thinking latency/tokens bounded so short CV answers never truncate."""
    normalized = model.strip().lower()
    if normalized.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    if normalized.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)
    return None


def _build_contents(
    history: list[dict[str, str]],
    user_message: str,
) -> list[types.Content]:
    contents = [
        types.Content(
            role="model" if turn["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=turn["content"])],
        )
        for turn in history
    ]
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    )
    return contents


def _build_config(model: str, system_prompt: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.3,
        thinking_config=_thinking_config_for(model),
    )


def _extract_visible_text(response: types.GenerateContentResponse) -> str:
    candidates = response.candidates or []
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return ""
    return "".join(
        part.text or ""
        for part in candidates[0].content.parts
        if not getattr(part, "thought", False)
    )


def _raise_mapped_error(exc: Exception) -> NoReturn:
    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            raise LLMRateLimitError("AI provider rate limit reached. Try again shortly.") from exc
        if exc.code in (408, 504):
            raise LLMTimeoutError("The AI provider timed out. Please try again.") from exc
        raise LLMUpstreamError(f"AI provider returned an error (HTTP {exc.code}).") from exc
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        raise LLMTimeoutError("The AI provider timed out. Please try again.") from exc
    raise LLMUpstreamError("Could not reach the AI provider.") from exc


def generate_reply(
    api_key: str,
    model: str,
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
) -> str:
    """Generate a single complete reply from Google AI Studio."""
    contents = _build_contents(history, user_message)
    config = _build_config(model, system_prompt)

    try:
        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ) as client:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
    except (errors.APIError, httpx.HTTPError, TimeoutError, OSError) as exc:
        _raise_mapped_error(exc)

    text = _extract_visible_text(response).strip()
    return text or "Sorry, I could not generate a response."


def stream_reply(
    api_key: str,
    model: str,
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
) -> Iterator[str]:
    """Stream visible reply text deltas from Google AI Studio."""
    contents = _build_contents(history, user_message)
    config = _build_config(model, system_prompt)

    try:
        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ) as client:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                delta = _extract_visible_text(chunk)
                if delta:
                    yield delta
    except (errors.APIError, httpx.HTTPError, TimeoutError, OSError) as exc:
        _raise_mapped_error(exc)
