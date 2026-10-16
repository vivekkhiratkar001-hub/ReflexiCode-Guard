"""Member 1 — GitHub integration and PR review automation."""

from .client import GitHubClient
from .collector import PullRequestCollector, prepare_review_inputs
from .config import GitHubAppConfig, build_config
from .exceptions import (
    GitHubAPIError,
    GitHubAuthError,
    GitHubConfigError,
    GitHubIntegrationError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)
from .models import FileChange, InlineComment, PullRequestInfo, ReviewSummary
from .orchestrator import ReviewOrchestrator

__all__ = [
    "FileChange",
    "GitHubAPIError",
    "GitHubAppConfig",
    "GitHubAuthError",
    "GitHubClient",
    "GitHubConfigError",
    "GitHubIntegrationError",
    "GitHubNotFoundError",
    "GitHubRateLimitError",
    "InlineComment",
    "PullRequestCollector",
    "PullRequestInfo",
    "ReviewOrchestrator",
    "ReviewSummary",
    "build_config",
    "prepare_review_inputs",
]
