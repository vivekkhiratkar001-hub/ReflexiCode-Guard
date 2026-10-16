"""Command-line entry point for Member 1 GitHub PR review."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.llm.mock_provider import MockProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.pipeline import ReviewPipeline

from .adapters import select_analysis_provider, select_rules_provider
from .client import GitHubClient
from .config import GitHubAppConfig, build_config
from .exceptions import GitHubConfigError, GitHubIntegrationError
from .orchestrator import ReviewOrchestrator, comments_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a GitHub pull request, run the Member 3 review pipeline, and optionally publish comments.",
    )
    parser.add_argument("--owner", help="GitHub repository owner")
    parser.add_argument("--repo", help="GitHub repository name")
    parser.add_argument("--pr-number", type=int, help="Pull request number")
    parser.add_argument("--token", help="GitHub token (defaults to GITHUB_TOKEN)")
    parser.add_argument("--api-url", help="GitHub API base URL")
    parser.add_argument("--event-path", help="Path to a GitHub Actions event JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline and print comments without publishing (default unless --publish is set)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Create GitHub review comments. Requires a token with pull-request write access.",
    )
    parser.add_argument(
        "--use-mock-llm",
        action="store_true",
        help="Use MockProvider instead of Ollama (for tests and CI without a local model)",
    )
    parser.add_argument(
        "--mock-llm-response",
        help='JSON string returned by MockProvider, default: {"findings": []}',
    )
    parser.add_argument("--rules-file", help="Optional JSON file of repository rules (Member 2 stand-in)")
    parser.add_argument(
        "--analysis-file",
        help="Optional JSON file of static-analysis findings (Member 2 stand-in)",
    )
    return parser


def build_pipeline(config: GitHubAppConfig) -> ReviewPipeline:
    if config.use_mock_llm:
        return ReviewPipeline(MockProvider(config.mock_llm_response))
    return ReviewPipeline(OllamaProvider())


def write_github_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    path = Path(summary_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(markdown)
        if not markdown.endswith("\n"):
            handle.write("\n")


def run_from_config(config: GitHubAppConfig) -> int:
    token = config.token
    if not config.dry_run:
        token = config.require_token(for_write=True)
    client = GitHubClient(token=token, api_url=config.api_url)
    orchestrator = ReviewOrchestrator(
        client,
        build_pipeline(config),
        rules_provider=select_rules_provider(config.rules_file),
        analysis_provider=select_analysis_provider(config.analysis_file),
        dry_run=config.dry_run,
    )
    summary = orchestrator.run(config.owner, config.repo, config.pr_number)
    markdown = summary.to_markdown()
    print(markdown)
    print("\n### Comment preview\n")
    print(comments_preview(summary.intended_comments))
    write_github_step_summary(markdown)
    if summary.errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run and args.publish:
            print("Both --dry-run and --publish were set; using dry-run for safety.", file=sys.stderr)
        config = build_config(
            owner=args.owner,
            repo=args.repo,
            pr_number=args.pr_number,
            token=args.token,
            api_url=args.api_url,
            dry_run=True if args.dry_run or (args.dry_run and args.publish) else (
                False if args.publish else None
            ),
            use_mock_llm=True if args.use_mock_llm else None,
            mock_llm_response=args.mock_llm_response,
            rules_file=args.rules_file,
            analysis_file=args.analysis_file,
            event_path=args.event_path,
            publish=bool(args.publish and not args.dry_run),
        )
        return run_from_config(config)
    except GitHubConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except GitHubIntegrationError as exc:
        print(f"GitHub integration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
