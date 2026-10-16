"""Errors raised by the GitHub integration (Member 1)."""


class GitHubIntegrationError(Exception):
    """Base error for Member 1 GitHub integration failures."""


class GitHubConfigError(GitHubIntegrationError):
    """Raised when required configuration is missing or invalid."""


class GitHubAPIError(GitHubIntegrationError):
    """Raised when the GitHub HTTP API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubAuthError(GitHubAPIError):
    """Raised when authentication fails (missing, invalid, or expired token)."""


class GitHubNotFoundError(GitHubAPIError):
    """Raised when a repository or pull request cannot be found."""


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub rate limits are exhausted after retries."""
