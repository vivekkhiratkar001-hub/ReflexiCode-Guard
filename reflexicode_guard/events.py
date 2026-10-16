"""Day 1 (30 Sep): PR event handling + branch strategy gate."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

HANDLED_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}
ISSUE_RE = re.compile(r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)")


@dataclass
class PREvent:
    repo: str
    number: int
    action: str
    head_sha: str
    base_ref: str
    head_ref: str
    draft: bool
    from_fork: bool


def load_event(path: Optional[str] = None) -> dict:
    path = path or os.environ.get("GITHUB_EVENT_PATH")
    if not path:
        raise ValueError("No event file: pass --event or set GITHUB_EVENT_PATH")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_pr_event(payload: dict) -> PREvent:
    """Extract the fields we need from a `pull_request` webhook payload."""
    pr = payload.get("pull_request")
    if not pr:
        raise ValueError("Payload has no 'pull_request' key (not a PR event)")
    base_repo = pr["base"]["repo"]["full_name"]
    head_repo = (pr["head"].get("repo") or {}).get("full_name", base_repo)
    return PREvent(
        repo=payload.get("repository", {}).get("full_name", base_repo),
        number=pr["number"],
        action=payload.get("action", ""),
        head_sha=pr["head"]["sha"],
        base_ref=pr["base"]["ref"],
        head_ref=pr["head"]["ref"],
        draft=bool(pr.get("draft", False)),
        from_fork=head_repo != base_repo,
    )


def should_review(ev: PREvent, protected_bases=("main", "master", "develop")) -> tuple[bool, str]:
    """Branch strategy gate: decide whether this event deserves a review run."""
    if ev.action not in HANDLED_ACTIONS:
        return False, f"action '{ev.action}' is not handled"
    if ev.draft:
        return False, "draft PR (reviewed once marked ready_for_review)"
    if ev.base_ref not in protected_bases:
        return False, f"base branch '{ev.base_ref}' is not in {list(protected_bases)}"
    if ev.from_fork:
        return True, "fork PR: token is read-only, will fall back to dry-run/log output"
    return True, "ok"


def extract_linked_issues(body: str | None) -> list[int]:
    return sorted({int(n) for n in ISSUE_RE.findall(body or "")})
