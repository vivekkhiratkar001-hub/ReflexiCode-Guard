"""Thin GitHub REST API client.

This module only performs HTTP communication. Business logic lives in collector,
mapper, and publisher classes so tests can mock this layer independently.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urljoin

import requests

from .exceptions import GitHubAPIError, GitHubAuthError, GitHubNotFoundError, GitHubRateLimitError


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        start = section.find("<")
        end = section.find(">")
        if start != -1 and end != -1:
            return section[start + 1 : end]
    return None


class GitHubClient:
    """Authenticated (or anonymous) GitHub REST client with pagination and retries."""

    def __init__(
        self,
        token: str | None = None,
        *,
        api_url: str = "https://api.github.com",
        session: requests.Session | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/") + "/"
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleeper or time.sleep
        self._token = token.strip() if token and token.strip() else None
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "ReflexiCode-Guard-Member1",
            }
        )
        if self._token:
            self.session.headers["Authorization"] = f"Bearer {self._token}"

    def _redact(self, value: str) -> str:
        if self._token:
            return value.replace(self._token, "[REDACTED]")
        return value

    def _url(self, path: str) -> str:
        return urljoin(self.api_url, path.lstrip("/"))

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = GitHubAPIError(self._redact(f"GitHub request failed: {exc}"))
                continue

            if response.status_code in {401, 403} and "bad credentials" in response.text.lower():
                raise GitHubAuthError(
                    "GitHub authentication failed. Check GITHUB_TOKEN.",
                    status_code=response.status_code,
                )
            if response.status_code == 401:
                raise GitHubAuthError(
                    "GitHub authentication failed (HTTP 401).",
                    status_code=401,
                )
            if response.status_code == 404:
                raise GitHubNotFoundError(
                    "GitHub resource not found. Check the owner, repository, and pull request number.",
                    status_code=404,
                )
            if self._is_rate_limited(response) and attempt < self.max_retries:
                self._sleep(self._retry_after_seconds(response))
                continue
            if self._is_rate_limited(response):
                raise GitHubRateLimitError(
                    "GitHub rate limit exceeded. Wait and retry later.",
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                detail = self._redact(response.text[:400].strip())
                reason = self._redact(response.reason)
                raise GitHubAPIError(
                    f"GitHub API error HTTP {response.status_code}: {detail or reason}",
                    status_code=response.status_code,
                )
            return response

        raise last_error or GitHubAPIError("GitHub request failed after retries.")

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            return True
        return "rate limit" in response.text.lower()

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                delay = int(reset) - int(time.time())
                return float(max(1, min(delay, 60)))
            except ValueError:
                pass
        return 1.0

    def _iter_pages(self, path: str, params: dict[str, Any] | None = None) -> Iterator[Any]:
        query = {"per_page": 100}
        if params:
            query.update(params)
        url: str | None = self._url(path)
        first = True
        while url:
            response = self._request("GET", url, params=query if first else None)
            first = False
            try:
                payload = response.json()
            except ValueError as exc:
                raise GitHubAPIError("GitHub returned a non-JSON response.") from exc
            if isinstance(payload, list):
                yield from payload
            else:
                yield payload
            url = _next_link(response.headers.get("Link"))

    def get_pull_request(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        response = self._request("GET", self._url(f"repos/{owner}/{repo}/pulls/{number}"))
        try:
            payload = response.json()
        except ValueError as exc:
            raise GitHubAPIError("GitHub returned a non-JSON pull request payload.") from exc
        if not isinstance(payload, dict):
            raise GitHubAPIError("Unexpected pull request payload from GitHub.")
        return payload

    def list_pull_request_files(self, owner: str, repo: str, number: int) -> list[dict[str, Any]]:
        items = list(self._iter_pages(f"repos/{owner}/{repo}/pulls/{number}/files"))
        return [item for item in items if isinstance(item, dict)]

    def list_review_comments(self, owner: str, repo: str, number: int) -> list[dict[str, Any]]:
        items = list(self._iter_pages(f"repos/{owner}/{repo}/pulls/{number}/comments"))
        return [item for item in items if isinstance(item, dict)]

    def create_pull_review(
        self,
        owner: str,
        repo: str,
        number: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._request(
            "POST",
            self._url(f"repos/{owner}/{repo}/pulls/{number}/reviews"),
            json=payload,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise GitHubAPIError("GitHub returned a non-JSON review response.") from exc
        if not isinstance(body, dict):
            raise GitHubAPIError("Unexpected review-creation payload from GitHub.")
        return body
