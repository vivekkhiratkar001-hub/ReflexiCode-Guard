"""Ollama-backed implementation of the code-review provider."""

import requests

from .context import ReviewContext
from .prompts import ReviewPromptBuilder
from .provider import LLMProvider


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

		try:
			response = requests.post(
				self.api_url,
				json=payload,
				timeout=self.timeout,
			)
			response.raise_for_status()
		except requests.exceptions.ConnectionError as exc:
			raise RuntimeError(
				f"Could not connect to Ollama at {self.api_url}. "
				"Make sure Ollama is running."
			) from exc
		except requests.exceptions.Timeout as exc:
			raise RuntimeError(
				f"The Ollama request timed out after {self.timeout} seconds."
			) from exc
		except requests.exceptions.HTTPError as exc:
			status = exc.response.status_code if exc.response is not None else "unknown"
			body = exc.response.text[:500] if exc.response is not None else ""
			detail = f" Details: {body}" if body else ""
			raise RuntimeError(f"Ollama returned HTTP {status}.{detail}") from exc
		except requests.exceptions.RequestException as exc:
			raise RuntimeError(f"The Ollama request failed: {exc}") from exc

		try:
			result = response.json()
		except ValueError as exc:
			raise RuntimeError("Ollama returned an invalid JSON response.") from exc

		if not isinstance(result, dict):
			raise RuntimeError("Ollama returned a JSON response with an unexpected format.")
		review_text = result.get("response")
		if not isinstance(review_text, str):
			raise RuntimeError("Ollama's response did not contain review text.")
		return review_text
