"""Command-line entry point for Member 1 GitHub PR review."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.llm.mock_provider import MockProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.pipeline import ReviewPipeline
from src.llm.prompts import ReviewPromptBuilder

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
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        help="Ollama request timeout in seconds (defaults to REFLEXICODE_OLLAMA_TIMEOUT or 180)",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        help="Maximum Ollama output tokens (defaults to REFLEXICODE_OLLAMA_NUM_PREDICT or 768)",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        help="Ollama context tokens (defaults to REFLEXICODE_OLLAMA_NUM_CTX or 8192)",
    )
    parser.add_argument(
        "--review-prompt-chars",
        type=int,
        help="Total review prompt limit (default 10000)",
    )
    parser.add_argument(
        "--review-code-chars",
        type=int,
        help="Changed-code character budget (default 4000)",
    )
    parser.add_argument(
        "--review-analysis-chars",
        type=int,
        help="Analysis-evidence character budget (default 2500)",
    )
    parser.add_argument(
        "--review-max-files",
        type=int,
        help="Maximum changed files selected for semantic review (default 5)",
    )
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
    prompt_builder = ReviewPromptBuilder(
        max_prompt_chars=config.review_prompt_chars,
        code_chars=config.review_code_chars,
        analysis_chars=config.review_analysis_chars,
        max_files=config.review_max_files,
    )
    return ReviewPipeline(
        OllamaProvider(
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict,
            num_ctx=config.ollama_num_ctx,
            prompt_builder=prompt_builder,
        ),
        prompt_builder=prompt_builder,
    )


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
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
            ollama_timeout=args.ollama_timeout,
            ollama_num_predict=args.ollama_num_predict,
            ollama_num_ctx=args.ollama_num_ctx,
            review_prompt_chars=args.review_prompt_chars,
            review_code_chars=args.review_code_chars,
            review_analysis_chars=args.review_analysis_chars,
            review_max_files=args.review_max_files,
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
