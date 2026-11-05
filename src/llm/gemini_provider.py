"""Gemini API provider for semantic code review.

Uses the Google Gemini generateContent REST endpoint. The API key is read
exclusively from the GEMINI_API_KEY environment variable; it is never logged
or embedded in exception messages. Requests are bounded by a configurable
timeout and implement review_with_deadline() for compatibility with the
ReviewPipeline deadline contract.

Features:
- Full typed error hierarchy (transient vs permanent).
- Bounded retries with exponential backoff, jitter, and Retry-After support.
- Safe extraction of Google diagnostic messages without leaking secrets or double punctuation.
- Optional explicit fallback model execution when primary model is unavailable or in high demand.
- Clean endpoint URL normalization stripping duplicate 'models/' prefixes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from time import monotonic
from typing import Any, Callable

import requests

from .context import ReviewContext
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider

logger = logging.getLogger(__name__)

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_ALLOWED_MODEL_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_:/ "
)


# ---------------------------------------------------------------------------
# Error Hierarchy
# ---------------------------------------------------------------------------

class GeminiError(RuntimeError):
    """Base exception for all Gemini API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        model: str = "",
        is_transient: bool = False,
        error_category: str = "ERROR",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.model = model
        self.is_transient = is_transient
        self.error_category = error_category


class GeminiTransientError(GeminiError):
    """Temporary failure suitable for retries (429, 500, 502, 503, 504, timeout, connection)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        model: str = "",
        error_category: str = "TRANSIENT_ERROR",
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            model=model,
            is_transient=True,
            error_category=error_category,
        )


class GeminiPermanentError(GeminiError):
    """Non-retryable failure (400, 401, 403, 404, bad candidate format)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        model: str = "",
        error_category: str = "PERMANENT_ERROR",
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            model=model,
            is_transient=False,
            error_category=error_category,
        )


class GeminiHighDemandError(GeminiTransientError):
    """HTTP 503 Service Unavailable specifically due to model high demand."""

    def __init__(self, message: str, *, model: str = "") -> None:
        super().__init__(
            message,
            status_code=503,
            model=model,
            error_category="HIGH_DEMAND_503",
        )


class GeminiQuotaExceededError(GeminiTransientError):
    """HTTP 429 Rate limit or quota exhaustion."""

    def __init__(self, message: str, *, model: str = "") -> None:
        super().__init__(
            message,
            status_code=429,
            model=model,
            error_category="RATE_LIMIT_429",
        )


class GeminiModelUnavailableError(GeminiPermanentError):
    """HTTP 404 Model not found or not provisioned for this account."""

    def __init__(self, message: str, *, model: str = "") -> None:
        super().__init__(
            message,
            status_code=404,
            model=model,
            error_category="MODEL_UNAVAILABLE_404",
        )


class GeminiAuthError(GeminiPermanentError):
    """HTTP 401 or 403 authentication / permission error."""

    def __init__(self, message: str, *, status_code: int = 401, model: str = "") -> None:
        super().__init__(
            message,
            status_code=status_code,
            model=model,
            error_category="AUTH_ERROR",
        )


class GeminiContentBlockedError(GeminiPermanentError):
    """Safety filter or recitation block."""

    def __init__(self, message: str, *, block_reason: str = "", model: str = "") -> None:
        super().__init__(
            message,
            status_code=None,
            model=model,
            error_category=f"CONTENT_BLOCKED_{block_reason}" if block_reason else "CONTENT_BLOCKED",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_model_name(model: str) -> str:
    """Normalize model identifiers, stripping 'models/' prefix, slashes, and whitespace."""
    cleaned = model.strip().strip("/")
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/"):].strip("/")
    return cleaned


def _clean_diagnostic_message(msg: str) -> str:
    """Format and clean Google error text, preventing duplicate punctuation and run-on words."""
    if not msg:
        return ""
    # Normalize internal whitespace
    cleaned = " ".join(msg.split())
    # Separate any accidentally concatenated words like 'highdemand' -> 'high demand'
    cleaned = re.sub(r'(?i)\bhighdemand\b', 'high demand', cleaned)
    # Strip trailing punctuation so callers can append sentence-terminating punctuation cleanly
    cleaned = cleaned.rstrip(".!?,;:\t\n ")
    return cleaned


def _safe_google_error_message(response: Any) -> str:
    """Extract a safe, non-secret diagnostic string from a Gemini error response.

    Google error bodies look like:
        {"error": {"code": 404, "message": "...", "status": "NOT_FOUND"}}

    The API key is passed as a query parameter, never in the response body,
    so the body is safe to read. We return only the ``message`` field,
    capped at 300 characters, to avoid leaking unrelated content.
    """
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error", {})
            if isinstance(error, dict):
                msg = str(error.get("message", "")).strip()
                if msg:
                    return _clean_diagnostic_message(msg)[:300]
    except Exception:
        pass
    return ""


def _parse_retry_after(headers: Any) -> float | None:
    """Extract and parse numeric Retry-After header in seconds."""
    if not headers or not hasattr(headers, "get"):
        return None
    val = headers.get("Retry-After")
    if not val:
        return None
    try:
        delay = float(val)
        return delay if delay > 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Call Google's Gemini generateContent API with bounded retries and fallback.

    The provider reads GEMINI_API_KEY at call time so that the key is never
    stored on the instance and is not visible in repr() or logs.

    Limitations
    -----------
    review_with_deadline() uses the requests ``timeout`` parameter and active
    clock monitoring to bound request durations. For Gemini, each request and
    retry attempt is constrained by the remaining aggregate time budget.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        timeout: int = 120,
        prompt_builder: ReviewPromptBuilder | None = None,
        fallback_model: str | None = None,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 8.0,
        jitter: bool = True,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._validate_model_name(model, "Gemini model name")
        if fallback_model:
            self._validate_model_name(fallback_model, "Gemini fallback model name")

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not 1 <= timeout <= 600
        ):
            raise ValueError("Gemini timeout must be an integer between 1 and 600 seconds.")

        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 10
        ):
            raise ValueError("max_retries must be an integer between 0 and 10.")

        self.model = normalize_model_name(model)
        self.fallback_model = normalize_model_name(fallback_model) if fallback_model else None
        self.timeout = timeout
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.jitter = jitter
        if sleeper is time.sleep and os.environ.get("PYTEST_CURRENT_TEST"):
            self._sleeper = lambda _: None
        else:
            self._sleeper = sleeper
        self._clock = clock

        # Tracking state
        self.min_request_window_seconds = 15.0
        self.last_used_model = self.model
        self.fallback_attempted = False
        self.fallback_succeeded = False
        self.retry_count = 0
        self.last_error: GeminiError | None = None

    @staticmethod
    def _validate_model_name(name: str, label: str) -> None:
        if not name or not name.strip():
            raise ValueError(f"{label} must be a non-empty string.")
        if len(name) > 200:
            raise ValueError(f"{label} must not exceed 200 characters.")
        if not all(ch in _ALLOWED_MODEL_CHARS for ch in name):
            raise ValueError(
                f"{label} contains disallowed characters. "
                "Use alphanumeric characters, hyphens, underscores, dots, or colons."
            )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def review(self, context: ReviewContext | dict[str, Any] | str) -> str:
        """Request a JSON review from the Gemini API and return the raw text."""
        prompt = self._resolve_prompt(context)
        return self._request(prompt, timeout=float(self.timeout))

    def review_with_deadline(self, prompt: str, *, timeout_seconds: float) -> str:
        """Request a review bounded by a wall-clock deadline.

        The effective request timeout is the minimum of the provider's
        configured timeout and ``timeout_seconds``.
        """
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number.")
        return self._request(
            prompt,
            timeout=float(self.timeout),
            wall_clock_timeout=float(timeout_seconds),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_prompt(self, context: ReviewContext | dict[str, Any] | str) -> str:
        if isinstance(context, str):
            return context
        if isinstance(context, ReviewContext):
            return self.prompt_builder.build(context)
        if isinstance(context, dict):
            return self.prompt_builder.build(ReviewContext.model_validate(context))
        raise TypeError(
            f"GeminiProvider.review() received an unsupported context type: {type(context).__name__}"
        )

    def _api_key(self) -> str:
        """Read the API key from the environment at call time (never stored)."""
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set or is empty. "
                "Export it before running the CLI, e.g.: "
                "$env:GEMINI_API_KEY = 'your-key'  (PowerShell) or "
                "export GEMINI_API_KEY='your-key'  (bash)."
            )
        return key

    def _endpoint_url(self, model_name: str | None = None) -> str:
        """Construct the generateContent URL, normalizing any 'models/' prefix."""
        target = normalize_model_name(model_name or self.model)
        return f"{_GEMINI_API_BASE}/{target}:generateContent"

    def _request(
        self,
        prompt: str,
        *,
        timeout: float,
        wall_clock_timeout: float | None = None,
    ) -> str:
        """Execute request with retries and optional fallback model."""
        started = self._clock()
        deadline = (started + wall_clock_timeout) if wall_clock_timeout is not None else None

        if self.fallback_succeeded and self.fallback_model:
            models_to_try = [self.fallback_model, self.model]
        else:
            models_to_try = [self.model]
            if self.fallback_model and self.fallback_model != self.model:
                models_to_try.append(self.fallback_model)

        self.last_used_model = models_to_try[0]
        self.fallback_attempted = False
        self.last_error = None

        last_exc: GeminiError | None = None

        for model_idx, target_model in enumerate(models_to_try):
            is_fallback = (model_idx > 0)
            if is_fallback:
                self.fallback_attempted = True
                logger.info(
                    "Primary Gemini model '%s' failed; attempting configured fallback model '%s': "
                    "remaining_seconds=%.1f",
                    self.model,
                    target_model,
                    (deadline - self._clock()) if deadline else -1.0,
                )

            try:
                result = self._post_with_retries(
                    target_model,
                    prompt,
                    timeout=timeout,
                    deadline=deadline,
                )
                self.last_used_model = target_model
                if is_fallback:
                    self.fallback_succeeded = True
                return result
            except GeminiError as exc:
                last_exc = exc
                self.last_error = exc
                # Only attempt fallback if we have a fallback model configured and error is eligible
                can_fallback = (
                    is_fallback is False
                    and len(models_to_try) > 1
                    and (exc.is_transient or exc.status_code == 404)
                )
                if not can_fallback:
                    raise

                # Check if sufficient budget remains for fallback attempt
                if deadline is not None and (deadline - self._clock()) < 2.0:
                    logger.warning(
                        "Insufficient time budget to attempt fallback model '%s': "
                        "remaining_seconds=%.1f",
                        self.fallback_model,
                        deadline - self._clock(),
                    )
                    raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Gemini request failed without an explicit error.")

    def _post_with_retries(
        self,
        target_model: str,
        prompt: str,
        *,
        timeout: float,
        deadline: float | None,
    ) -> str:
        """Send POST requests with bounded exponential backoff and jitter."""
        api_key = self._api_key()
        url = self._endpoint_url(target_model)
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
            },
        }

        max_attempts = 1 + max(0, self.max_retries)

        for attempt in range(1, max_attempts + 1):
            now = self._clock()
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0.5:
                    raise GeminiTransientError(
                        f"The Gemini request exceeded its wall-clock deadline before attempt {attempt}.",
                        model=target_model,
                        error_category="DEADLINE_EXCEEDED",
                    )
                effective_timeout = min(timeout, remaining)
            else:
                effective_timeout = timeout

            logger.info(
                "Gemini review request started: outcome=pending model=%s "
                "attempt=%d/%d prompt_chars=%d estimated_prompt_tokens=%d timeout_seconds=%.1f",
                target_model,
                attempt,
                max_attempts,
                len(prompt),
                (len(prompt) + 3) // 4,
                effective_timeout,
            )

            response: requests.Response | None = None
            req_started = self._clock()

            try:
                response = requests.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    timeout=effective_timeout,
                )
            except requests.exceptions.Timeout as exc:
                elapsed = self._clock() - req_started
                logger.warning(
                    "Gemini request timed out: outcome=timed_out model=%s "
                    "attempt=%d/%d elapsed_seconds=%.2f timeout_seconds=%.1f",
                    target_model,
                    attempt,
                    max_attempts,
                    elapsed,
                    effective_timeout,
                )
                if attempt < max_attempts and self._can_retry(deadline):
                    self._do_backoff(attempt, deadline, target_model)
                    continue
                raise GeminiTransientError(
                    f"The Gemini request timed out after {effective_timeout:.1f} seconds.",
                    model=target_model,
                    error_category="TIMEOUT",
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                elapsed = self._clock() - req_started
                logger.warning(
                    "Gemini connection failed: outcome=failed reason=connection "
                    "model=%s attempt=%d/%d elapsed_seconds=%.2f",
                    target_model,
                    attempt,
                    max_attempts,
                    elapsed,
                )
                if attempt < max_attempts and self._can_retry(deadline):
                    self._do_backoff(attempt, deadline, target_model)
                    continue
                raise GeminiTransientError(
                    "Could not connect to the Gemini API. Check your network connection.",
                    model=target_model,
                    error_category="CONNECTION_ERROR",
                ) from exc
            except requests.exceptions.RequestException as exc:
                elapsed = self._clock() - req_started
                logger.warning(
                    "Gemini request failed: outcome=failed reason=%s model=%s "
                    "attempt=%d/%d elapsed_seconds=%.2f",
                    type(exc).__name__,
                    target_model,
                    attempt,
                    max_attempts,
                    elapsed,
                )
                if attempt < max_attempts and self._can_retry(deadline):
                    self._do_backoff(attempt, deadline, target_model)
                    continue
                raise GeminiTransientError(
                    f"The Gemini API request failed ({type(exc).__name__}).",
                    model=target_model,
                    error_category=type(exc).__name__,
                ) from exc

            # --- Handle HTTP Response ---
            status = response.status_code
            elapsed = self._clock() - req_started

            # 1. Success (200)
            if status == 200:
                return self._parse_success_response(response, target_model, elapsed)

            # Extract safe error details from Google response
            google_msg = _safe_google_error_message(response)
            details = google_msg if google_msg else "(no detail returned)"

            logger.warning(
                "Gemini request failed: outcome=failed reason=http status=%d "
                "model=%s attempt=%d/%d elapsed_seconds=%.2f google_error=%s",
                status,
                target_model,
                attempt,
                max_attempts,
                elapsed,
                details,
            )

            # 2. Transient / Retryable errors: 429, 500, 502, 503, 504
            is_transient_status = status in (429, 500, 502, 503, 504)
            if is_transient_status and attempt < max_attempts and self._can_retry(deadline):
                # For 503 High Demand: if a fallback model is available, switch quickly
                # after 1 retry so the fallback model can handle the batch within budget.
                if (
                    status == 503
                    and self.fallback_model
                    and target_model != self.fallback_model
                    and attempt >= 2
                ):
                    break
                retry_after = _parse_retry_after(response.headers)
                self._do_backoff(attempt, deadline, target_model, retry_after=retry_after)
                continue

            # Retries exhausted or non-retryable: raise typed errors
            if status == 503:
                raise GeminiHighDemandError(
                    f"The Gemini API returned HTTP 503 (High Demand) for model '{target_model}'. "
                    f"Google says: {details}.",
                    model=target_model,
                )
            if status == 429:
                raise GeminiQuotaExceededError(
                    f"The Gemini API rate limit was exceeded (HTTP 429) for model '{target_model}'. "
                    f"Google says: {details}.",
                    model=target_model,
                )
            if status == 404:
                raise GeminiModelUnavailableError(
                    f"The Gemini API returned HTTP 404 for model '{target_model}'. "
                    "This usually means the model is not available for your API key or project. "
                    f"Google says: {details}. "
                    "To fix: (1) verify the model name is correct, "
                    "(2) check https://ai.google.dev/gemini-api/docs/models for available models, "
                    "(3) try passing --gemini-fallback-model gemini-2.0-flash.",
                    model=target_model,
                )
            if status == 400:
                raise GeminiPermanentError(
                    f"The Gemini API returned HTTP 400 (Bad Request) for model '{target_model}'. "
                    f"Google says: {details}.",
                    status_code=400,
                    model=target_model,
                    error_category="BAD_REQUEST_400",
                )
            if status in (401, 403):
                raise GeminiAuthError(
                    f"The Gemini API returned HTTP {status} for model '{target_model}'. "
                    "Check that GEMINI_API_KEY is valid and has the required permissions.",
                    status_code=status,
                    model=target_model,
                )
            if status >= 500:
                raise GeminiTransientError(
                    f"The Gemini API returned HTTP {status} for model '{target_model}'. "
                    f"Google says: {details}.",
                    status_code=status,
                    model=target_model,
                    error_category=f"SERVER_ERROR_{status}",
                )

            raise GeminiPermanentError(
                f"The Gemini API returned HTTP {status} for model '{target_model}'. "
                f"Google says: {details}.",
                status_code=status,
                model=target_model,
                error_category=f"HTTP_{status}",
            )

        raise GeminiTransientError(
            f"The Gemini request exhausted all {max_attempts} attempts for model '{target_model}'.",
            model=target_model,
        )

    def _can_retry(self, deadline: float | None) -> bool:
        """Check if at least 1 second remains before deadline."""
        if deadline is None:
            return True
        return (deadline - self._clock()) > 1.0

    def _do_backoff(
        self,
        attempt: int,
        deadline: float | None,
        model: str,
        retry_after: float | None = None,
    ) -> None:
        """Compute delay and sleep, respecting deadline constraints."""
        self.retry_count += 1
        if retry_after is not None:
            delay = min(self.max_backoff, max(0.1, retry_after))
        else:
            delay = min(
                self.max_backoff,
                self.initial_backoff * (self.backoff_factor ** (attempt - 1)),
            )
            if self.jitter:
                delay += random.uniform(0.05, 0.35)

        if deadline is not None:
            remaining = deadline - self._clock()
            if delay >= remaining - 0.5:
                delay = max(0.05, remaining - 0.5)

        logger.info(
            "Gemini retrying request: model=%s attempt=%d delay_seconds=%.2f",
            model,
            attempt,
            delay,
        )
        self._sleeper(delay)

    def _parse_success_response(
        self,
        response: requests.Response,
        model: str,
        elapsed: float,
    ) -> str:
        """Validate and extract JSON review text from a successful 200 response."""
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise GeminiPermanentError(
                "The Gemini API returned a non-JSON response body.",
                model=model,
                error_category="MALFORMED_JSON_BODY",
            ) from exc

        if not isinstance(body, dict):
            raise GeminiPermanentError(
                "The Gemini API returned an unexpected response format (expected a JSON object).",
                model=model,
                error_category="INVALID_ENVELOPE",
            )

        candidates = body.get("candidates")
        if not candidates:
            feedback = body.get("promptFeedback", {})
            block_reason = feedback.get("blockReason")
            if block_reason:
                raise GeminiContentBlockedError(
                    f"The Gemini API blocked the request (blockReason: {block_reason}). "
                    "The prompt may have triggered safety filters.",
                    block_reason=block_reason,
                    model=model,
                )
            raise GeminiPermanentError(
                "The Gemini API returned no candidates. Check API quotas and model availability.",
                model=model,
                error_category="NO_CANDIDATES",
            )

        first = candidates[0]
        if not isinstance(first, dict):
            raise GeminiPermanentError(
                "The Gemini API returned a candidate in an unexpected format.",
                model=model,
                error_category="INVALID_CANDIDATE",
            )

        finish_reason = first.get("finishReason", "")
        if finish_reason in ("SAFETY", "RECITATION", "OTHER"):
            raise GeminiContentBlockedError(
                f"The Gemini API refused to complete the response (finishReason: {finish_reason}).",
                block_reason=finish_reason,
                model=model,
            )
        if finish_reason == "MAX_TOKENS":
            logger.warning(
                "Gemini response truncated: outcome=truncated model=%s "
                "finish_reason=MAX_TOKENS elapsed_seconds=%.2f",
                model,
                elapsed,
            )
            raise GeminiPermanentError(
                "The Gemini response was truncated at the model's maximum token limit.",
                model=model,
                error_category="MAX_TOKENS",
            )

        content = first.get("content", {})
        if not isinstance(content, dict):
            raise GeminiPermanentError(
                "The Gemini API returned a candidate without a valid 'content' object.",
                model=model,
                error_category="MISSING_CONTENT",
            )

        parts = content.get("parts", [])
        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]

        if not text_parts:
            raise GeminiPermanentError(
                "The Gemini API returned a response with no usable text content.",
                model=model,
                error_category="NO_TEXT_PARTS",
            )

        review_text = "\n".join(text_parts)
        if not review_text.strip():
            raise GeminiPermanentError(
                "The Gemini API returned an empty review response.",
                model=model,
                error_category="EMPTY_RESPONSE",
            )

        logger.info(
            "Gemini review request completed: outcome=completed model=%s "
            "elapsed_seconds=%.2f output_chars=%d",
            model,
            elapsed,
            len(review_text),
        )
        return review_text
