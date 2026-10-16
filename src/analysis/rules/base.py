"""Base definitions and data structures for static, security, and complexity rules."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuleFinding:
    """Internal representation of a rule finding before normalization."""
    rule_id: str
    category: str
    severity: str
    message: str
    suggestion: str
    line_number: Optional[int]
    evidence: str
