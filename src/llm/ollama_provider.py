"""Ollama-backed implementation of the code-review provider."""

import json

import requests

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

	@staticmethod
	def _format_section(context: dict, key: str) -> str:
		"""Format text or structured context for inclusion in the prompt."""
		value = context.get(key)
		if value is None or value == "":
			return "Not provided."
		if isinstance(value, str):
			return value
		return json.dumps(value, indent=2, ensure_ascii=False)

	def _build_prompt(self, context: dict) -> str:
		"""Build a focused, evidence-based code-review prompt."""
		sections = (
			("PR title", "pr_title"),
			("PR description", "pr_description"),
			("Issue context", "issue_context"),
			("Repository rules", "repository_rules"),
			("Changed code", "changed_code"),
			("Existing AST/static-analysis findings", "analysis_findings"),
		)
		provided_context = "\n\n".join(
			f"## {title}\n{self._format_section(context, key)}"
			for title, key in sections
		)

		return f"""You are a careful semantic code reviewer. Review only the changed code.
Use the PR and issue context to understand the change's intent. Use repository rules
and existing analysis findings as supporting evidence, not as automatic proof.

Never invent files, line numbers, repository rules, or evidence. Report only issues
supported by the supplied context. Explain each issue concisely and actionably, and
suggest a correction when possible. Return concise code-review findings only. If no
evidence-supported issue is present, say that no findings were identified.

{provided_context}
"""

	def review(self, context: dict) -> str:
		"""Ask Ollama to review the context and return its response text."""
		payload = {
			"model": self.model,
			"prompt": self._build_prompt(context),
			"stream": False,
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
			raise RuntimeError(f"Ollama returned HTTP {status}.") from exc
		except requests.exceptions.RequestException as exc:
			raise RuntimeError(f"The Ollama request failed: {exc}") from exc

		try:
			result = response.json()
		except ValueError as exc:
			raise RuntimeError("Ollama returned an invalid JSON response.") from exc

		review_text = result.get("response")
		if not isinstance(review_text, str):
			raise RuntimeError("Ollama's response did not contain review text.")
		return review_text
