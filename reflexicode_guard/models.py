"""Shared data contracts.

`Finding` is THE interface between all four members' modules:
  M1 (rules)  -> source="rule"
  M2 (AST)    -> source="ast" | "static" | "security"
  M3 (LLM)    -> source="llm"
  M4 (this)   -> validates target lines, de-duplicates, publishes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional

SEVERITIES = ("info", "low", "medium", "high", "critical")
SOURCES = ("rule", "ast", "static", "security", "llm")


@dataclass(frozen=True)
class Finding:
    file: str                 # repo-relative path, e.g. "src/app.py"
    line: int                 # line number in the NEW (head) version of the file
    severity: str             # one of SEVERITIES
    category: str             # e.g. "security", "style", "logic"
    message: str              # human-readable explanation
    evidence: str             # code snippet / rule text justifying the finding
    source: str = "static"    # one of SOURCES
    rule_id: str = ""         # e.g. "SEC001" or "CONTRIBUTING#naming"
    confidence: float = 1.0

    def validate(self) -> list[str]:
        errs = []
        if not self.file:
            errs.append("file is empty")
        if not isinstance(self.line, int) or self.line < 1:
            errs.append("line must be a positive int")
        if self.severity not in SEVERITIES:
            errs.append(f"severity must be one of {SEVERITIES}")
        if self.source not in SOURCES:
            errs.append(f"source must be one of {SOURCES}")
        if not self.message.strip():
            errs.append("message is empty")
        if not self.evidence.strip():
            errs.append("evidence is required (evidence-gated schema)")
        return errs

    @property
    def fingerprint(self) -> str:
        """Stable id so re-runs on `synchronize` do not repost the same comment."""
        raw = f"{self.file}|{self.line}|{self.rule_id}|{self.category}|{self.message}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        return d

    @staticmethod
    def from_dict(d: dict) -> "Finding":
        d = {k: v for k, v in d.items() if k != "fingerprint"}
        return Finding(**d)


@dataclass
class ChangedFile:
    path: str
    status: str                       # added / modified / removed / renamed
    patch: Optional[str] = None       # unified-diff hunks (None for binary / huge files)
    additions: int = 0
    deletions: int = 0


@dataclass
class PRContext:
    repo: str                         # "owner/name"
    number: int
    title: str
    body: str
    author: str
    base_ref: str
    head_ref: str
    base_sha: str
    head_sha: str                     # the "current commit" reviews must anchor to
    draft: bool = False
    labels: list[str] = field(default_factory=list)
    files: list[ChangedFile] = field(default_factory=list)
    linked_issues: list[int] = field(default_factory=list)

    def file(self, path: str) -> Optional[ChangedFile]:
        return next((f for f in self.files if f.path == path), None)
