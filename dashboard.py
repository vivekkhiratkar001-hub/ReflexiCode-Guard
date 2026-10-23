"""Interactive Streamlit dashboard for GitHub pull-request code reviews."""

from __future__ import annotations

import html
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import requests
import streamlit as st

from src.github.adapters import select_analysis_provider, select_rules_provider
from src.github.cli import build_pipeline
from src.github.client import GitHubClient
from src.github.config import build_config
from src.github.exceptions import (
    GitHubAPIError,
    GitHubIntegrationError,
    GitHubRateLimitError,
)
from src.github.models import InlineComment, ReviewSummary
from src.github.orchestrator import ReviewOrchestrator
from src.shared.models import AnalysisFinding


SEVERITIES = ("Critical", "High", "Medium", "Low")
SEVERITY_COLORS = {
    "critical": "#b42318",
    "high": "#d92d20",
    "medium": "#b54708",
    "low": "#175cd3",
}


def _severity_key(value: str) -> str:
    return value.strip().casefold()


def _finding_category(index: int, summary: ReviewSummary) -> str:
    return "Static Analysis" if index < summary.static_findings else "Semantic Bug"


def _render_finding(finding: AnalysisFinding, category: str) -> None:
    severity = finding.severity.title()
    color = SEVERITY_COLORS.get(_severity_key(severity), "#475467")
    st.markdown(
        f'<span style="background:{color};color:white;padding:3px 9px;'
        f'border-radius:12px;font-weight:600">{html.escape(severity)}</span>'
        f'&nbsp; **{html.escape(category)}**',
        unsafe_allow_html=True,
    )
    st.text(f"File: {finding.file_path} · Line: {finding.line_number or 'N/A'}")
    st.text(finding.message)
    if finding.evidence:
        st.caption("Code evidence")
        st.code(finding.evidence)
    if finding.suggestion:
        st.caption("Proposed suggestion / fix")
        st.code(finding.suggestion, language="diff")


def _render_findings(summary: ReviewSummary) -> None:
    all_findings = summary.findings
    filter_columns = st.columns(2)
    with filter_columns[0]:
        chosen_severities = st.multiselect(
            "Filter by Severity",
            options=list(SEVERITIES),
            default=list(SEVERITIES),
            key="finding_severity_filter",
        )
    with filter_columns[1]:
        chosen_categories = st.multiselect(
            "Filter by Category",
            options=["Static Analysis", "Semantic Bug"],
            default=["Static Analysis", "Semantic Bug"],
            key="finding_category_filter",
        )

    filtered = [
        (index, item)
        for index, item in enumerate(all_findings)
        if item.severity.title() in chosen_severities
        and _finding_category(index, summary) in chosen_categories
    ]
    st.caption(f"Showing {len(filtered)} of {len(all_findings)} findings")
    if not filtered:
        st.info("No findings match the selected filters.")
        return

    for index, (finding_index, finding) in enumerate(filtered, start=1):
        label = f"Finding {index} · {finding.severity.title()}"
        with st.expander(label, expanded=index == 1):
            _render_finding(finding, _finding_category(finding_index, summary))


def _render_comments(comments: list[InlineComment]) -> None:
    if not comments:
        st.info("No inline comments would be published for this review.")
        return
    for index, comment in enumerate(comments, start=1):
        side_description = (
            "RIGHT (added/current file side)"
            if comment.side == "RIGHT"
            else "LEFT (deleted/old file side)"
        )
        with st.expander(
            f"Inline comment {index} · {side_description}",
            expanded=index == 1,
        ):
            st.text(f"{comment.path}:{comment.line} · {comment.side}")
            st.caption(f"Fingerprint: `{comment.fingerprint}`")
            st.code(comment.body, language="markdown")


def _render_coverage(summary: ReviewSummary) -> None:
    st.subheader("Reviewed files")
    if summary.reviewed_files:
        for path in summary.reviewed_files:
            st.text(f"✅ {path}")
    else:
        st.info("No files were fully reviewed.")

    if summary.partially_reviewed_files:
        st.subheader("Partially reviewed files")
        for path in summary.partially_reviewed_files:
            st.text(f"🟡 {path}")

    st.subheader("Unreviewed / omitted files")
    if summary.unreviewed_files:
        for path in summary.unreviewed_files:
            st.text(f"⚠️ {path}")
        st.warning(
            "These files were prioritized out because of the configured file and "
            "context budgets. Increase Max Files Budget or the review budget and rerun "
            "to expand semantic coverage."
        )
    else:
        st.success("No files were omitted from semantic review.")

    if summary.skipped_files:
        st.subheader("Files not eligible for review")
        for path in summary.skipped_files:
            st.text(f"⏭️ {path}")
        st.caption("These files were skipped because the diff was binary, missing, or unsupported.")

    if summary.warnings:
        for warning in summary.warnings:
            st.warning(warning)

    st.subheader("LLM generation stats")
    stats = st.columns(3)
    stats[0].metric("Batches", summary.semantic_batches_total)
    stats[1].metric("Failed batches", summary.semantic_batches_failed)
    stats[2].metric("Elapsed", f"{summary.semantic_elapsed_seconds:.2f}s")


def _render_summary(summary: ReviewSummary) -> None:
    st.markdown(summary.to_markdown())
    payload = json.dumps(asdict(summary), indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download review_summary.json",
        data=payload,
        file_name="review_summary.json",
        mime="application/json",
        key="download_review_json",
    )
    st.download_button(
        "⬇️ Download summary.md",
        data=summary.to_markdown(),
        file_name="summary.md",
        mime="text/markdown",
        key="download_review_markdown",
    )


def _run_review(
    owner: str,
    repo: str,
    pr_number: int,
    token: str | None,
    model: str,
    timeout: int,
    max_files: int,
    use_mock_llm: bool,
    dry_run: bool,
) -> ReviewSummary:
    config = build_config(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        token=token,
        ollama_model=model,
        ollama_timeout=timeout,
        review_max_files=max_files,
        use_mock_llm=use_mock_llm,
        dry_run=dry_run,
        publish=not dry_run,
    )
    client = GitHubClient(token=config.token, api_url=config.api_url)
    orchestrator = ReviewOrchestrator(
        client,
        build_pipeline(config),
        rules_provider=select_rules_provider(config.rules_file),
        analysis_provider=select_analysis_provider(config.analysis_file),
        dry_run=config.dry_run,
    )
    return orchestrator.run(config.owner, config.repo, config.pr_number)


def _run_button_action(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    token: str | None,
    model: str,
    timeout: int,
    max_files: int,
    use_mock_llm: bool,
    dry_run: bool,
) -> None:
    if not dry_run:
        st.warning(
            "Dry Run is off. If the GitHub token has pull-request write access, "
            "this review may publish comments."
        )

    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflexicode-review") as executor:
            future = executor.submit(
                _run_review,
                owner,
                repo,
                pr_number,
                token,
                model,
                timeout,
                max_files,
                use_mock_llm,
                dry_run,
            )
            with st.status("Review running · 0s elapsed", expanded=True) as status:
                while not future.done():
                    elapsed = int(time.monotonic() - started)
                    status.update(
                        label=f"Review running · {elapsed}s elapsed",
                        state="running",
                    )
                    time.sleep(0.25)
                result = future.result()
                status.update(
                    label=f"Review finished · {time.monotonic() - started:.1f}s",
                    state="complete",
                )
    except (
        GitHubIntegrationError,
        requests.exceptions.RequestException,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        _display_error(error)
        return

    st.session_state["review_summary"] = result


def _display_error(error: Exception) -> None:
    if isinstance(error, GitHubRateLimitError):
        st.error(f"GitHub rate limit reached: {error}")
        st.info("Wait for the rate-limit window to reset, or use an authenticated token.")
    elif isinstance(error, GitHubAPIError):
        st.error(f"GitHub API request failed: {error}")
        st.info("Check the owner, repository, PR number, token permissions, and API availability.")
    elif isinstance(error, GitHubIntegrationError):
        st.error(f"GitHub review configuration failed: {error}")
        st.info("Check the repository details and set GITHUB_TOKEN for private repositories.")
    elif isinstance(error, requests.exceptions.ConnectionError):
        st.error(f"Could not connect to GitHub or the local Ollama service: {error}")
        st.info("Check network access and, unless using Mock LLM, confirm Ollama is running locally.")
    elif isinstance(error, requests.exceptions.Timeout):
        st.error(f"A GitHub or Ollama request timed out: {error}")
        st.info("Increase the timeout or check that the requested service is responsive.")
    elif isinstance(error, requests.exceptions.HTTPError):
        st.error(f"GitHub or Ollama returned an unsuccessful HTTP response: {error}")
        st.info("Check repository/token access and verify the selected Ollama model is installed.")
    elif isinstance(error, requests.exceptions.RequestException):
        st.error(f"GitHub or Ollama request failed: {error}")
        st.info("Check service availability, network access, and repository permissions.")
    elif isinstance(error, ValueError):
        st.error(f"Invalid review configuration: {error}")
        st.info("Check the model name, timeout, and file-budget values.")
    else:
        st.error(f"Review failed: {error}")


def main() -> None:
    st.set_page_config(page_title="ReflexiCode-Guard Dashboard", page_icon="🛡️", layout="wide")
    st.title("🛡️ ReflexiCode-Guard Review Dashboard")
    st.caption("Run a bounded static and semantic review of a GitHub pull request.")

    with st.sidebar:
        st.header("Repository settings")
        owner = st.text_input("Owner", value="vivekkhiratkar001-hub")
        repo = st.text_input("Repo", value="ReflexiCode-Guard")
        pr_number = st.number_input("PR Number", min_value=1, value=3, step=1)
        token = st.text_input(
            "GitHub Token",
            value=os.environ.get("GITHUB_TOKEN", ""),
            type="password",
            help="Optional for public repositories; GITHUB_TOKEN is used when present.",
        )

        st.header("LLM Settings")
        model_options = [
            "qwen2.5-coder:7b",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:3b",
            "Custom...",
        ]
        model_choice = st.selectbox("Ollama Model", model_options)
        model = (
            st.text_input("Custom Ollama Model", value="qwen2.5-coder:7b")
            if model_choice == "Custom..."
            else model_choice
        )
        timeout = st.slider("Ollama Timeout (seconds)", 30, 300, 180)
        max_files = st.slider("Max Files Budget", 1, 20, 8)
        use_mock_llm = st.checkbox("Use Mock LLM", value=False)

        st.header("Mode")
        dry_run = st.checkbox("Dry Run", value=True)
        run_clicked = st.button("▶ Run Code Review", type="primary", use_container_width=True)

    if run_clicked:
        if not owner.strip() or not repo.strip():
            st.error("Owner and Repo must not be empty.")
        elif model_choice == "Custom..." and not model.strip():
            st.error("Enter a model name or choose one of the listed models.")
        else:
            _run_button_action(
                owner=owner.strip(),
                repo=repo.strip(),
                pr_number=int(pr_number),
                token=token.strip() or None,
                model=model.strip(),
                timeout=timeout,
                max_files=max_files,
                use_mock_llm=use_mock_llm,
                dry_run=dry_run,
            )

    summary = st.session_state.get("review_summary")
    if not isinstance(summary, ReviewSummary):
        st.info("Configure a pull request in the sidebar and run a review to see results.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Findings", summary.total_findings)
    severity_column = metric_columns[1]
    severity_column.markdown("**Findings by Severity**")
    for severity in SEVERITIES:
        count = next(
            (
                number
                for name, number in summary.findings_by_severity.items()
                if _severity_key(name) == _severity_key(severity)
            ),
            0,
        )
        color = SEVERITY_COLORS[_severity_key(severity)]
        severity_column.markdown(
            f'<span style="color:{color};font-weight:700">{severity}: {count}</span>',
            unsafe_allow_html=True,
        )
    semantic_status = (
        "clean" if summary.semantic_status == "success_no_findings" else summary.semantic_status
    )
    metric_columns[2].metric("Semantic Status", semantic_status)
    reviewed_count = len(summary.reviewed_files) + len(summary.partially_reviewed_files)
    total_scope = (
        reviewed_count
        + len(summary.unreviewed_files)
        + len(summary.skipped_files)
    )
    metric_columns[3].metric(
        "Reviewed Files",
        f"{reviewed_count} of {total_scope}",
    )

    if summary.errors:
        for error in summary.errors:
            st.error(error)
        if any("ollama" in error.casefold() or "model" in error.casefold() for error in summary.errors):
            st.info("Check that Ollama is running, the selected model is installed, and the timeout is sufficient.")
        elif any("rate limit" in error.casefold() for error in summary.errors):
            st.info("Wait for GitHub's rate-limit window to reset, or authenticate with a GitHub token.")
        elif any("github" in error.casefold() or "token" in error.casefold() for error in summary.errors):
            st.info("Check the repository/PR details and verify the token has the required repository permissions.")

    findings_tab, comments_tab, scope_tab, export_tab = st.tabs(
        [
            "📋 All Findings",
            "💬 GitHub Inline Comments Preview",
            "📊 Review Scope & File Coverage",
            "📝 Markdown Summary & JSON Export",
        ]
    )
    with findings_tab:
        _render_findings(summary)
    with comments_tab:
        _render_comments(summary.intended_comments)
    with scope_tab:
        _render_coverage(summary)
    with export_tab:
        _render_summary(summary)


if __name__ == "__main__":
    main()
