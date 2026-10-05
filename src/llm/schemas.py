"""Pydantic schemas used by the LLM review module."""

from typing import Optional

from pydantic import BaseModel


class ReviewFinding(BaseModel):
	"""A single evidence-supported code-review finding."""

	file: str
	line: int
	category: str
	severity: str
	message: str
	evidence: str
	suggestion: Optional[str] = None
