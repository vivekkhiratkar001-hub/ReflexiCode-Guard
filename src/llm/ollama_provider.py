"""Ollama-backed implementation of the code-review provider."""

import json
import logging
import re
from time import monotonic
from typing import Any

import requests

from .context import ReviewContext
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider
from .schemas import ReviewFindings

logger = logging.getLogger(__name__)


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
    ) -> None:
        if not 1 <= timeout <= 300:
            raise ValueError("timeout must be between 1 and 300 seconds.")
        if not 64 <= num_predict <= 2048 or not 2048 <= num_ctx <= 32768:
            raise ValueError(
                "num_predict must be 64-2048 and num_ctx must be 2048-32768."
            )
        if num_predict >= num_ctx:
            raise ValueError("num_predict must be positive and smaller than num_ctx.")
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.num_predict = num_predict
        self.num_ctx = num_ctx
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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "options": {
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }

        started = monotonic()
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
            self.timeout,
            self.num_predict,
            self.num_ctx,
        )
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            for raw_chunk in response.iter_lines(decode_unicode=True):
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
            state = "before first output" if first_output_at is None else "after first output"
            first_output = (
                "not received"
                if first_output_at is None
                else f"{first_output_at - started:.2f} seconds"
            )
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
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Ollama request failed: outcome=failed reason=connection elapsed_seconds=%.2f.",
                monotonic() - started,
            )
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            logger.warning(
                "Ollama request failed: outcome=failed reason=http status=%s elapsed_seconds=%.2f.",
                status,
                monotonic() - started,
            )
            raise RuntimeError(f"Ollama returned HTTP {status}.") from exc
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Ollama request failed: outcome=failed reason=%s elapsed_seconds=%.2f.",
                type(exc).__name__,
                monotonic() - started,
            )
            raise RuntimeError(
                f"The Ollama request failed ({type(exc).__name__})."
            ) from exc
        finally:
            if response is not None:
                response.close()

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
