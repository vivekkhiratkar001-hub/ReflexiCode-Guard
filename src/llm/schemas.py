"""Validated data models used by the LLM review module."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


class ReviewFinding(BaseModel):
	"""A single evidence-supported, line-level code-review finding."""

	model_config = ConfigDict(extra="forbid", strict=True)

	file: StrictStr = Field(min_length=1)
	line: StrictInt = Field(gt=0)
	category: StrictStr = Field(min_length=1)
	severity: Literal["Low", "Medium", "High", "Critical"]
	message: StrictStr = Field(min_length=1)
	evidence: StrictStr = Field(min_length=1)
	suggestion: StrictStr | None = None

	@field_validator("severity", mode="before")
	@classmethod
	def normalize_severity(cls, value: object) -> object:
		"""Accept case variations of the allowed severity labels."""
		if isinstance(value, str):
			stripped = value.strip().casefold()
			for level in ("low", "medium", "high", "critical"):
				if stripped == level:
					return level.capitalize()
		return value

	@field_validator("file", "category", "message", "evidence")
	@classmethod
	def reject_whitespace_only(cls, value: str) -> str:
		"""Require useful non-whitespace content in each required text field."""
		value = value.strip()
		if not value:
			raise ValueError("must not be empty or whitespace-only")
		return value


class ReviewFindings(BaseModel):
	"""The JSON envelope returned by the model."""

	model_config = ConfigDict(extra="forbid", strict=True)

	findings: list[ReviewFinding]
