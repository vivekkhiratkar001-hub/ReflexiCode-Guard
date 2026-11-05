"""Command-line entry point for Member 1 GitHub PR review."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.llm.gemini_provider import GeminiProvider
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
        "--ollama-model",
        help="Installed local Ollama model (default qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        help="Ollama per-request timeout in seconds (default REFLEXICODE_OLLAMA_TIMEOUT or 120)",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        help="Maximum Ollama output tokens (default REFLEXICODE_OLLAMA_NUM_PREDICT or 512)",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        help="Ollama context tokens (default REFLEXICODE_OLLAMA_NUM_CTX or 4096)",
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
        help="Maximum changed files per review batch (default 3)",
    )
    parser.add_argument(
        "--review-max-findings",
        type=int,
        help="Maximum semantic findings requested per batch (default 3)",
    )
    parser.add_argument(
        "--review-max-requests",
        type=int,
        help="Maximum sequential LLM requests per PR review (default 5)",
    )
    parser.add_argument(
        "--review-budget-seconds",
        type=int,
        help="Aggregate semantic-review scheduling budget in seconds (default 240)",
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
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "gemini"],
        help="LLM backend to use for semantic review (default: ollama). "
             "Can also be set via REFLEXICODE_LLM_PROVIDER environment variable.",
    )
    parser.add_argument(
        "--gemini-model",
        help="Gemini model name (default: gemini-2.0-flash). "
             "Can also be set via REFLEXICODE_GEMINI_MODEL environment variable.",
    )
    parser.add_argument(
        "--gemini-fallback-model",
        help="Optional fallback Gemini model used if the primary model fails or experiences high demand. "
             "Can also be set via REFLEXICODE_GEMINI_FALLBACK_MODEL environment variable.",
    )
    parser.add_argument(
        "--gemini-max-retries",
        type=int,
        help="Maximum retries for transient Gemini errors (default: 3). "
             "Can also be set via REFLEXICODE_GEMINI_MAX_RETRIES environment variable.",
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
        max_findings=config.review_max_findings,
    )
    if config.llm_provider == "gemini":
        provider = GeminiProvider(
            model=config.gemini_model,
            timeout=config.ollama_timeout,  # reuse the shared timeout budget
            prompt_builder=prompt_builder,
            fallback_model=config.gemini_fallback_model,
            max_retries=config.gemini_max_retries if config.gemini_max_retries is not None else 3,
        )
    else:
        # Default: Ollama
        provider = OllamaProvider(
            model=config.ollama_model,
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict,
            num_ctx=config.ollama_num_ctx,
            prompt_builder=prompt_builder,
        )
    return ReviewPipeline(
        provider,
        prompt_builder=prompt_builder,
        max_requests=config.review_max_requests,
        review_budget_seconds=config.review_budget_seconds,
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
        rules_provider=select_rules_provider(config.rules_file, client=client),
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
            ollama_model=args.ollama_model,
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
            review_max_findings=args.review_max_findings,
            review_max_requests=args.review_max_requests,
            review_budget_seconds=args.review_budget_seconds,
            event_path=args.event_path,
            publish=bool(args.publish and not args.dry_run),
            llm_provider=args.llm_provider,
            gemini_model=args.gemini_model,
            gemini_fallback_model=args.gemini_fallback_model,
            gemini_max_retries=args.gemini_max_retries,
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
