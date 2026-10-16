"""Configuration for GitHub integration.

Required values can come from CLI flags, environment variables, or a GitHub
Actions event payload. No secrets are stored in code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .exceptions import GitHubConfigError


DEFAULT_API_URL = "https://api.github.com"
DEFAULT_OLLAMA_TIMEOUT = 180


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GitHubAppConfig:
    """Runtime settings for collecting a PR and optionally publishing comments."""

    owner: str
    repo: str
    pr_number: int
    token: str | None
    api_url: str
    dry_run: bool
    use_mock_llm: bool
    mock_llm_response: str
    rules_file: Path | None
    analysis_file: Path | None
    ollama_timeout: int = DEFAULT_OLLAMA_TIMEOUT
    head_sha: str | None = None

    def require_token(self, *, for_write: bool) -> str:
        if self.token:
            return self.token
        if for_write:
            raise GitHubConfigError(
                "A GitHub token is required to publish review comments. "
                "Set GITHUB_TOKEN or pass --token."
            )
        raise GitHubConfigError(
            "A GitHub token is required to read this pull request. "
            "Set GITHUB_TOKEN or pass --token. "
            "Public repositories may work without a token, but authenticated "
            "requests are recommended."
        )


def _split_repository(value: str) -> tuple[str, str]:
    parts = value.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise GitHubConfigError(
            f"Repository must be in owner/repo form, got {value!r}."
        )
    return parts[0], parts[1]


def load_github_event(event_path: Path) -> dict:
    try:
        payload = json.loads(event_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GitHubConfigError(f"GitHub event file not found: {event_path}") from exc
    except json.JSONDecodeError as exc:
        raise GitHubConfigError(f"GitHub event file is not valid JSON: {event_path}") from exc
    if not isinstance(payload, dict):
        raise GitHubConfigError("GitHub event payload must be a JSON object.")
    return payload


def config_from_event(payload: dict) -> tuple[str, str, int, str | None]:
    """Extract owner, repo, PR number, and head SHA from an Actions event."""
    repository = payload.get("repository") or {}
    full_name = repository.get("full_name") or os.environ.get("GITHUB_REPOSITORY", "")
    pull = payload.get("pull_request") or {}
    number = pull.get("number") or payload.get("number")
    head = pull.get("head") or {}
    sha = head.get("sha") if isinstance(head, dict) else None
    if not full_name or number is None:
        raise GitHubConfigError(
            "The GitHub event payload does not contain repository.full_name and a pull request number."
        )
    try:
        pr_number = int(number)
    except (TypeError, ValueError) as exc:
        raise GitHubConfigError(f"Invalid pull request number in event payload: {number!r}.") from exc
    owner, repo = _split_repository(str(full_name))
    return owner, repo, pr_number, str(sha) if sha else None


def build_config(
    *,
    owner: str | None = None,
    repo: str | None = None,
    pr_number: int | None = None,
    token: str | None = None,
    api_url: str | None = None,
    dry_run: bool | None = None,
    use_mock_llm: bool | None = None,
    mock_llm_response: str | None = None,
    rules_file: str | None = None,
    analysis_file: str | None = None,
    ollama_timeout: int | None = None,
    event_path: str | None = None,
    publish: bool = False,
) -> GitHubAppConfig:
    """Merge CLI arguments with environment variables and optional event JSON."""
    event_owner = event_repo = None
    event_pr: int | None = None
    event_sha: str | None = None
    resolved_event = event_path or os.environ.get("GITHUB_EVENT_PATH")
    if resolved_event:
        payload = load_github_event(Path(resolved_event))
        event_owner, event_repo, event_pr, event_sha = config_from_event(payload)

    env_repo = os.environ.get("GITHUB_REPOSITORY")
    env_owner = env_repo_name = None
    if env_repo:
        env_owner, env_repo_name = _split_repository(env_repo)

    resolved_owner = owner or event_owner or env_owner or os.environ.get("GITHUB_OWNER")
    resolved_repo = repo or event_repo or env_repo_name or os.environ.get("GITHUB_REPO")
    resolved_pr = pr_number if pr_number is not None else event_pr
    if resolved_pr is None and os.environ.get("GITHUB_PR_NUMBER"):
        try:
            resolved_pr = int(os.environ["GITHUB_PR_NUMBER"])
        except ValueError as exc:
            raise GitHubConfigError(
                f"GITHUB_PR_NUMBER must be an integer, got {os.environ['GITHUB_PR_NUMBER']!r}."
            ) from exc

    if not resolved_owner or not resolved_repo or resolved_pr is None:
        raise GitHubConfigError(
            "Missing pull-request identity. Provide --owner, --repo, and --pr-number "
            "(or GITHUB_REPOSITORY plus GITHUB_PR_NUMBER, or a GitHub Actions event file)."
        )
    if int(resolved_pr) <= 0:
        raise GitHubConfigError(f"Pull request number must be positive, got {resolved_pr}.")

    resolved_token = token if token is not None else os.environ.get("GITHUB_TOKEN")
    if resolved_token is not None:
        resolved_token = resolved_token.strip() or None

    raw_timeout = os.environ.get("REFLEXICODE_OLLAMA_TIMEOUT")
    try:
        timeout_value = (
            ollama_timeout
            if ollama_timeout is not None
            else int(raw_timeout) if raw_timeout is not None else DEFAULT_OLLAMA_TIMEOUT
        )
    except ValueError as exc:
        raise GitHubConfigError(
            "REFLEXICODE_OLLAMA_TIMEOUT must be an integer between 1 and 900 seconds."
        ) from exc
    if timeout_value < 1 or timeout_value > 900:
        raise GitHubConfigError(
            "Ollama timeout must be an integer between 1 and 900 seconds."
        )

    # A non-publishing run is always a dry run. An environment variable must
    # never silently enable GitHub writes; publication requires the explicit
    # `publish=True` gate (wired to the CLI's --publish option).
    resolved_dry_run = not publish or dry_run is True

    resolved_mock = (
        use_mock_llm
        if use_mock_llm is not None
        else _truthy(os.environ.get("REFLEXICODE_USE_MOCK_LLM"), default=False)
    )
    resolved_mock_response = mock_llm_response or os.environ.get(
        "REFLEXICODE_MOCK_LLM_RESPONSE",
        '{"findings": []}',
    )
    return GitHubAppConfig(
        owner=resolved_owner,
        repo=resolved_repo,
        pr_number=int(resolved_pr),
        token=resolved_token,
        api_url=(api_url or os.environ.get("GITHUB_API_URL") or DEFAULT_API_URL).rstrip("/"),
        dry_run=resolved_dry_run,
        use_mock_llm=resolved_mock,
        mock_llm_response=resolved_mock_response,
        rules_file=Path(rules_file) if rules_file else (
            Path(os.environ["REFLEXICODE_RULES_FILE"])
            if os.environ.get("REFLEXICODE_RULES_FILE")
            else None
        ),
        analysis_file=Path(analysis_file) if analysis_file else (
            Path(os.environ["REFLEXICODE_ANALYSIS_FILE"])
            if os.environ.get("REFLEXICODE_ANALYSIS_FILE")
            else None
        ),
        ollama_timeout=timeout_value,
        head_sha=event_sha,
    )
