import os
"""Ollama-backed implementation of the code-review provider."""

import json
import logging
import math
import re
from threading import Event, Lock, Timer
from time import monotonic
from typing import Any

import requests

from .context import ReviewContext
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .schemas import ReviewFindings

logger = logging.getLogger(__name__)


def _is_complete_json_object(text: str) -> bool:
    """Check whether received output already forms one complete JSON object."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        _, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    return not stripped[end:].strip()



class _RequestDeadlineController:
    """Helper to manage request deadline timer and response cleanup."""

    def __init__(self, wall_clock_timeout: float | None = None) -> None:
        self.wall_clock_timeout = wall_clock_timeout
        self.response_holder: dict[str, Any] = {}
        self._cleaned = False
        self._lock = Lock()

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True
            resp = self.response_holder.get("response")
            if resp is not None and hasattr(resp, "close"):
                try:
                    resp.close()
                except Exception:
                    pass


class OllamaProvider(LLMProvider):
    """Stream a bounded review request from a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        api_url: str = "http://localhost:11434/api/generate",
        timeout: int = 120,
        num_predict: int = 512,
        num_ctx: int = 4096,
        prompt_builder: ReviewPromptBuilder | None = None,
        keep_alive: str = "30m",
    ) -> None:
        if not 1 <= timeout <= 300:
            raise ValueError("timeout must be between 1 and 300 seconds.")
        if not 64 <= num_predict <= 2048 or not 2048 <= num_ctx <= 32768:
            raise ValueError(
                "num_predict must be 64-2048 and num_ctx must be 2048-32768."
            )
        if num_predict >= num_ctx:
            raise ValueError("num_predict must be positive and smaller than num_ctx.")
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("keep_alive must be a non-empty string.")
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive.strip()
        self.prompt_builder = prompt_builder or ReviewPromptBuilder()

    def review(self, context: ReviewContext | dict[str, Any] | str) -> str:
        """Request a bounded JSON review and return its complete response text."""
        if isinstance(context, str):
            prompt = context
        else:
            review_context = (
                context
                if isinstance(context, ReviewContext)
                else ReviewContext.model_validate(context)
            )
            prompt = self.prompt_builder.build(review_context)
        return self._request(prompt)

    def review_with_deadline(self, prompt: str, *, timeout_seconds: float) -> str:
        """Request a prompt with a best-effort wall-clock deadline.

        Requests' timeout controls socket inactivity, not total elapsed time.
        For deadline-aware pipeline calls, a timer closes the active response;
        a monotonic check between stream records and the Requests timeout are
        additional guards. Closing the client response cannot guarantee that
        the Ollama server immediately stops its underlying model computation.
        """
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive.")
        return self._request(prompt, wall_clock_timeout=float(timeout_seconds))

    def _request(self, prompt: str, wall_clock_timeout: float | None = None) -> str:
        """Send one prompt and validate the transport-level completion marker."""
        options: dict[str, Any] = {
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }
        num_thread_env = os.environ.get("REFLEXICODE_OLLAMA_NUM_THREAD")
        if num_thread_env:
            try:
                options["num_thread"] = int(num_thread_env)
            except ValueError:
                pass
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": options,
        }

        started = monotonic()
        deadline = (
            started + wall_clock_timeout if wall_clock_timeout is not None else None
        )
        effective_timeout = (
            min(self.timeout, wall_clock_timeout)
            if wall_clock_timeout is not None
            else self.timeout
        )
        deadline_expired = Event()
        response_closed = Event()
        close_lock = Lock()
        response_holder: dict[str, Any] = {}
        deadline_timer: Timer | None = None

        def close_response(active_response: Any) -> None:
            with close_lock:
                if response_closed.is_set():
                    return
                active_response.close()
                response_closed.set()

        def expire_request() -> None:
            deadline_expired.set()
            active_response = response_holder.get("response")
            if active_response is not None:
                try:
                    close_response(active_response)
                except Exception:
                    # The caller observes the deadline event and reports the
                    # timeout; cleanup must not raise from the timer thread.
                    pass

        def wall_deadline_reached() -> bool:
            return deadline_expired.is_set() or (
                deadline is not None and monotonic() >= deadline
            )

        if wall_clock_timeout is not None:
            deadline_timer = Timer(wall_clock_timeout, expire_request)
            deadline_timer.daemon = True
            deadline_timer.start()
        first_output_at: float | None = None
        response = None
        fragments: list[str] = []
        final_record: dict[str, Any] | None = None
        logger.info(
            "Ollama review request started: outcome=pending model=%s prompt_chars=%d "
            "estimated_prompt_tokens=%d timeout_seconds=%s num_predict=%d num_ctx=%d",
            self.model,
            len(prompt),
            (len(prompt) + 3) // 4,
            effective_timeout,
            self.num_predict,
            self.num_ctx,
        )
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=effective_timeout,
                stream=True,
            )
            response_holder["response"] = response
            if wall_deadline_reached():
                raise RuntimeError(
                    f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                    "wall-clock deadline before response streaming began."
                )
            response.raise_for_status()
            for raw_chunk in response.iter_lines(decode_unicode=True):
                if wall_deadline_reached():
                    raise RuntimeError(
                        f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                        "wall-clock deadline while streaming."
                    )
                if not raw_chunk:
                    continue
                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise RuntimeError(
                            "Ollama returned a streaming response that was not valid UTF-8."
                        ) from exc
                try:
                    chunk = json.loads(raw_chunk)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Ollama returned malformed streaming JSON; the response is incomplete."
                    ) from exc
                if not isinstance(chunk, dict):
                    raise RuntimeError(
                        "Ollama returned a streaming chunk with an unexpected format."
                    )
                text = chunk.get("response", "")
                if not isinstance(text, str):
                    raise RuntimeError(
                        "Ollama returned a streaming chunk without valid review text."
                    )
                if text:
                    fragments.append(text)
                    if first_output_at is None:
                        first_output_at = monotonic()
                if chunk.get("done") is True:
                    final_record = chunk
                    break
        except RuntimeError:
            if wall_deadline_reached():
                logger.warning(
                    "Ollama wall-clock deadline expired: outcome=timed_out "
                    "elapsed_seconds=%.2f deadline_seconds=%.2f.",
                    monotonic() - started,
                    wall_clock_timeout,
                )
            else:
                logger.warning(
                    "Ollama response stream rejected: outcome=invalid_output elapsed_seconds=%.2f "
                    "first_output=%s.",
                    monotonic() - started,
                    (
                        f"{first_output_at - started:.2f} seconds"
                        if first_output_at is not None
                        else "not received"
                    ),
                )
            raise
        except requests.exceptions.Timeout as exc:
            elapsed = monotonic() - started
            if wall_deadline_reached():
                logger.warning(
                    "Ollama wall-clock deadline expired: outcome=timed_out model=%s "
                    "elapsed_seconds=%.2f deadline_seconds=%.2f first_output=%s",
                    self.model,
                    elapsed,
                    wall_clock_timeout,
                    "not received" if first_output_at is None else "received",
                )
                raise RuntimeError(
                    f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                    "wall-clock deadline."
                ) from exc
            state = "before first output" if first_output_at is None else "after first output"
            first_output = (
                "not received"
                if first_output_at is None
                else f"{first_output_at - started:.2f} seconds"
            )
            # A stalled stream may have already delivered the complete JSON
            # object before the final "done" frame. Keep evidence validation
            # strict by only salvaging output that is a complete JSON object;
            # truncated output still fails closed.
            salvaged = "".join(fragments).strip()
            if _is_complete_json_object(salvaged):
                logger.warning(
                    "Ollama request timed out but complete JSON was received: "
                    "outcome=completed_after_timeout elapsed_seconds=%.2f "
                    "first_output_seconds=%s output_chars=%d.",
                    elapsed,
                    (
                        f"{first_output_at - started:.2f}"
                        if first_output_at is not None
                        else "none"
                    ),
                    len(salvaged),
                )
                return salvaged
            logger.warning(
                "Ollama request timed out: outcome=timed_out model=%s elapsed_seconds=%.2f "
                "first_output=%s state=%s",
                self.model,
                elapsed,
                first_output,
                state,
            )
            raise RuntimeError(
                f"The Ollama request timed out after {self.timeout} seconds {state}."
            ) from exc
        except AttributeError as exc:
            # The deadline timer may close the response while iter_lines() is
            # blocked waiting for data, leaving the underlying connection as
            # None. Treat this as a wall-clock deadline expiry rather than
            # crashing the review run.
            if wall_deadline_reached():
                logger.warning(
                    "Ollama wall-clock deadline expired: outcome=timed_out "
                    "elapsed_seconds=%.2f deadline_seconds=%.2f.",
                    monotonic() - started,
                    wall_clock_timeout,
                )
                raise RuntimeError(
                    f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                    "wall-clock deadline."
                ) from exc
            raise
        except requests.exceptions.ConnectionError as exc:
            if wall_deadline_reached():
                logger.warning(
                    "Ollama wall-clock deadline expired: outcome=timed_out "
                    "elapsed_seconds=%.2f deadline_seconds=%.2f.",
                    monotonic() - started,
                    wall_clock_timeout,
                )
                raise RuntimeError(
                    f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                    "wall-clock deadline."
                ) from exc
            logger.warning(
                "Ollama request failed: outcome=failed reason=connection elapsed_seconds=%.2f.",
                monotonic() - started,
            )
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            if wall_deadline_reached():
                logger.warning(
                    "Ollama wall-clock deadline expired: outcome=timed_out "
                    "elapsed_seconds=%.2f deadline_seconds=%.2f.",
                    monotonic() - started,
                    wall_clock_timeout,
                )
                raise RuntimeError(
                    f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                    "wall-clock deadline."
                ) from exc
            status = exc.response.status_code if exc.response is not None else "unknown"
            logger.warning(
                "Ollama request failed: outcome=failed reason=http status=%s elapsed_seconds=%.2f.",
                status,
                monotonic() - started,
            )
            raise RuntimeError(f"Ollama returned HTTP {status}.") from exc
        except requests.exceptions.RequestException as exc:
            if wall_deadline_reached():
                logger.warning(
                    "Ollama wall-clock deadline expired: outcome=timed_out "
                    "elapsed_seconds=%.2f deadline_seconds=%.2f.",
                    monotonic() - started,
                    wall_clock_timeout,
                )
                raise RuntimeError(
                    f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                    "wall-clock deadline."
                ) from exc
            logger.warning(
                "Ollama request failed: outcome=failed reason=%s elapsed_seconds=%.2f.",
                type(exc).__name__,
                monotonic() - started,
            )
            raise RuntimeError(
                f"The Ollama request failed ({type(exc).__name__})."
            ) from exc
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()
            if response is not None:
                close_response(response)
            if deadline_timer is not None and deadline_timer.is_alive():
                deadline_timer.join(timeout=0.05)

        if wall_deadline_reached():
            raise RuntimeError(
                f"The Ollama request exceeded its {wall_clock_timeout:.1f}-second "
                "wall-clock deadline during response cleanup."
            )

        elapsed = monotonic() - started
        first_output_seconds = (
            first_output_at - started if first_output_at is not None else None
        )
        if final_record is None:
            logger.warning(
                "Ollama response stream incomplete: outcome=invalid_output elapsed_seconds=%.2f.",
                elapsed,
            )
            raise RuntimeError(
                "Ollama response stream ended before completion; the response may be truncated."
            )
        if final_record.get("done_reason") == "length":
            logger.warning(
                "Ollama response truncated: outcome=invalid_output num_predict=%d elapsed_seconds=%.2f.",
                self.num_predict,
                elapsed,
            )
            raise RuntimeError(
                "Ollama response was truncated at the configured generation token limit."
            )

        review_text = "".join(fragments)
        if not review_text.strip():
            logger.warning(
                "Ollama response empty: outcome=invalid_output elapsed_seconds=%.2f.",
                elapsed,
            )
            raise RuntimeError("Ollama returned an empty review response.")

        logger.info(
                "Ollama review request completed: outcome=completed elapsed_seconds=%.2f "
            "first_output_seconds=%s output_chars=%d generated_tokens=%s "
            "prompt_eval_seconds=%s generation_seconds=%s",
            elapsed,
            f"{first_output_seconds:.2f}" if first_output_seconds is not None else "none",
            len(review_text),
            final_record.get("eval_count", "unknown"),
            self._duration_seconds(final_record.get("prompt_eval_duration")),
            self._duration_seconds(final_record.get("eval_duration")),
        )
        return review_text

    @staticmethod
    def _duration_seconds(value: Any) -> str:
        if not isinstance(value, (int, float)) or value < 0:
            return "unknown"
        return f"{value / 1_000_000_000:.2f}"
