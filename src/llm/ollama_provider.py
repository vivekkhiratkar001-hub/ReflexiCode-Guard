"""Ollama-backed implementation of the code-review provider."""

import logging
from time import monotonic

import requests

from .context import ReviewContext
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
	"""Send pull-request review context to a local Ollama model."""

	def __init__(
		self,
		model: str = "qwen2.5-coder:7b",
		api_url: str = "http://localhost:11434/api/generate",
		timeout: int = 120,
	) -> None:
		self.model = model
		self.api_url = api_url
		self.timeout = timeout
		self.prompt_builder = ReviewPromptBuilder()

	def review(self, context: ReviewContext | dict | str) -> str:
		"""Ask Ollama to review a context or prompt and return raw JSON text."""
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
			"stream": False,
			"format": "json",
		}

		started = monotonic()
		logger.info(
			"Ollama review request started: prompt_chars=%d prompt_bytes=%d timeout_seconds=%s",
			len(prompt),
			len(prompt.encode("utf-8")),
			self.timeout,
		)
		try:
			response = requests.post(
				self.api_url,
				json=payload,
				timeout=self.timeout,
			)
			response.raise_for_status()
		except requests.exceptions.ConnectionError as exc:
			logger.warning("Ollama review request could not connect after %.2f seconds.", monotonic() - started)
			raise RuntimeError(
				"Could not connect to Ollama. Make sure Ollama is running."
			) from exc
		except requests.exceptions.Timeout as exc:
			logger.warning("Ollama review request timed out after %.2f seconds.", monotonic() - started)
			raise RuntimeError(
				f"The Ollama request timed out after {self.timeout} seconds."
			) from exc
		except requests.exceptions.HTTPError as exc:
			status = exc.response.status_code if exc.response is not None else "unknown"
			logger.warning("Ollama returned HTTP %s after %.2f seconds.", status, monotonic() - started)
			raise RuntimeError(f"Ollama returned HTTP {status}.") from exc
		except requests.exceptions.RequestException as exc:
			logger.warning("Ollama review request failed after %.2f seconds.", monotonic() - started)
			raise RuntimeError(
				f"The Ollama request failed ({type(exc).__name__})."
			) from exc

		try:
			result = response.json()
		except ValueError as exc:
			logger.warning("Ollama returned invalid JSON after %.2f seconds.", monotonic() - started)
			raise RuntimeError("Ollama returned an invalid JSON response.") from exc

		if not isinstance(result, dict):
			logger.warning("Ollama returned an unexpected response format after %.2f seconds.", monotonic() - started)
			raise RuntimeError("Ollama returned a JSON response with an unexpected format.")
		review_text = result.get("response")
		if not isinstance(review_text, str):
			logger.warning("Ollama response omitted review text after %.2f seconds.", monotonic() - started)
			raise RuntimeError("Ollama's response did not contain review text.")
		if not review_text.strip():
			logger.warning("Ollama returned an empty review after %.2f seconds.", monotonic() - started)
			raise RuntimeError("Ollama returned an empty review response.")
		logger.info("Ollama review request completed in %.2f seconds.", monotonic() - started)
		return review_text
