"""Interactive Streamlit dashboard for GitHub pull-request code reviews."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

# Ensure repository root is on sys.path when launched directly via streamlit
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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

_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def _check_ollama_model(model: str, timeout: float = 4.0) -> tuple[bool, str]:
    """Return (available, message) for the given Ollama model name.

    Queries the local Ollama /api/tags endpoint to determine whether the
    named model is installed. Returns a descriptive message for both cases.
    Does not raise; all network errors are caught and reported as unavailable.
    """
    try:
        resp = requests.get(_OLLAMA_TAGS_URL, timeout=timeout)
        resp.raise_for_status()
        tags = resp.json()
        installed = [m.get("name", "") for m in tags.get("models", [])]
        # Match either exact name or, if the query has no tag, any variant
        has_tag = ":" in model
        for name in installed:
            if has_tag:
                # Exact match required when tag is specified (e.g. "qwen2.5-coder:3b")
                if name == model:
                    return True, f"✅ `{model}` is installed in Ollama."
            else:
                # Bare base name (e.g. "qwen2.5-coder") matches any tag variant
                if name == model or name.startswith(model + ":"):
                    return True, f"✅ `{model}` is installed in Ollama."
        return (
            False,
            f"⚠️ `{model}` is **not installed** in Ollama. "
            f"Run `ollama pull {model}` before starting a review. "
            f"Installed models: {', '.join(installed) or 'none'}.",
        )
    except requests.exceptions.ConnectionError:
        return False, "⚠️ Ollama is not reachable at localhost:11434. Start Ollama before running a review."
    except requests.exceptions.Timeout:
        return False, "⚠️ Ollama health check timed out. Ollama may be starting or overloaded."
    except Exception as exc:
        return False, f"⚠️ Ollama health check failed: {exc}"


SEVERITIES = ("Critical", "High", "Medium", "Low")
SEVERITY_COLORS = {
    "critical": "#b42318",
    "high": "#d92d20",
    "medium": "#b54708",
    "low": "#175cd3",
}


def _severity_key(value: str) -> str:
    return value.strip().casefold()


def _finding_from_comment(comment: InlineComment) -> AnalysisFinding:
    body = (comment.body or "").strip()
    file_path = comment.finding_file or comment.path
    line_number = comment.finding_line or comment.line
    severity = "Medium"
    category = "Semantic"
    message = ""
    evidence = ""
    suggestion = ""

    # Strip GitHub fingerprint comment marker
    body_clean = re.sub(r"\n*<!--\s*reflexicode-guard:fp=[^>]*-->\s*$", "", body).strip()

    header_pattern = r"^\*\*\[(?P<severity>[^\]]+)\]\s*(?P<category>[^*]+)\*\*(?:\s+in\s+`(?P<file>[^:`]+)(?::(?P<line>\d+))?`)?"
    match = re.match(header_pattern, body_clean)
    if match:
        severity = match.group("severity").strip()
        category = match.group("category").strip()
        if match.group("file"):
            file_path = match.group("file").strip()
        if match.group("line"):
            try:
                line_number = int(match.group("line"))
            except ValueError:
                pass
        rest = body_clean[match.end():].strip()
    else:
        rest = body_clean

    evidence_match = re.search(r"\*\*Evidence:\*\*\s*(.*?)(?=\n\n\*\*Suggestion:\*\*|\Z)", rest, re.DOTALL)
    suggestion_match = re.search(r"\*\*Suggestion:\*\*\s*(.*)", rest, re.DOTALL)

    if evidence_match:
        message = rest[:evidence_match.start()].strip()
        evidence_raw = evidence_match.group(1).strip()
        if evidence_raw.startswith("`") and evidence_raw.endswith("`") and len(evidence_raw) >= 2:
            evidence = evidence_raw[1:-1]
        elif evidence_raw.startswith("```") and evidence_raw.endswith("```"):
            evidence = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", evidence_raw)
            evidence = re.sub(r"\n?```$", "", evidence).strip()
        else:
            evidence = evidence_raw
    elif suggestion_match:
        message = rest[:suggestion_match.start()].strip()
    else:
        message = rest

    if suggestion_match:
        sugg_raw = suggestion_match.group(1).strip()
        if sugg_raw == "No suggestion provided.":
            suggestion = ""
        else:
            suggestion = sugg_raw

    return AnalysisFinding(
        file_path=file_path,
        line_number=line_number,
        category=category,
        severity=severity,
        message=message,
        evidence=evidence,
        suggestion=suggestion,
    )


def _finding_from_unmapped(text: str) -> AnalysisFinding:
    pattern = r"^(?P<file>[^:]+):(?P<line>\d+)\s*\((?P<category>[^;)]+)(?:;\s*(?P<note>[^)]+))?\)"
    match = re.match(pattern, text.strip())
    if match:
        file_path = match.group("file").strip()
        try:
            line_number: int | None = int(match.group("line"))
        except ValueError:
            line_number = None
        category = match.group("category").strip()
        note = match.group("note")
        msg = f"Unmapped finding: {note}" if note else "Finding on line outside reviewed diff patch."
    else:
        file_path = text
        line_number = None
        category = "Unmapped"
        msg = text

    return AnalysisFinding(
        file_path=file_path,
        line_number=line_number,
        category=category,
        severity="Medium",
        message=msg,
        evidence="",
        suggestion="",
    )


def _extract_findings(summary: ReviewSummary) -> list[AnalysisFinding]:
    if hasattr(summary, "findings") and getattr(summary, "findings") is not None:
        return list(summary.findings)

    findings: list[AnalysisFinding] = []
    for comment in getattr(summary, "intended_comments", []):
        findings.append(_finding_from_comment(comment))

    for unmapped in getattr(summary, "unmapped_findings", []):
        findings.append(_finding_from_unmapped(unmapped))

    return findings


def _finding_category(index: int, summary: ReviewSummary, finding: AnalysisFinding | None = None) -> str:
    if finding and finding.category:
        cat_lower = finding.category.strip().casefold()
        if cat_lower in {"static analysis", "static_analysis"}:
            return "Static Analysis"
        if cat_lower in {"semantic", "semantic bug", "bug", "correctness", "security"}:
            return "Semantic Bug"
    return "Static Analysis" if index < getattr(summary, "static_findings", 0) else "Semantic Bug"


def _render_finding(finding: AnalysisFinding, category: str) -> None:
    severity = (finding.severity or "Medium").title()
    color = SEVERITY_COLORS.get(_severity_key(severity), "#475467")
    st.markdown(
        f'<span style="background:{color};color:white;padding:3px 9px;'
        f'border-radius:12px;font-weight:600">{html.escape(severity)}</span>'
        f'&nbsp; **{html.escape(category)}**',
        unsafe_allow_html=True,
    )
    st.text(f"File: {finding.file_path} · Line: {finding.line_number or 'N/A'}")
    st.text(finding.message or "No details provided.")
    if finding.evidence:
        st.caption("Code evidence")
        st.code(finding.evidence)
    if finding.suggestion:
        st.caption("Proposed suggestion / fix")
        st.code(finding.suggestion, language="diff")


def _render_findings(summary: ReviewSummary) -> None:
    all_findings = _extract_findings(summary)
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
        if (item.severity or "Medium").title() in chosen_severities
        and _finding_category(index, summary, item) in chosen_categories
    ]
    st.caption(f"Showing {len(filtered)} of {len(all_findings)} findings")
    if not filtered:
        st.info("No findings match the selected filters.")
        return

    for index, (finding_index, finding) in enumerate(filtered, start=1):
        label = f"Finding {index} · {(finding.severity or 'Medium').title()}"
        with st.expander(label, expanded=index == 1):
            _render_finding(finding, _finding_category(finding_index, summary, finding))


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
        review_max_findings=min(8, max(5, max_files)),
        use_mock_llm=use_mock_llm,
        dry_run=dry_run,
        publish=not dry_run,
    )
    client = GitHubClient(token=config.token, api_url=config.api_url)
    orchestrator = ReviewOrchestrator(
        client,
        build_pipeline(config),
        rules_provider=select_rules_provider(config.rules_file, client=client),
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


def _render_model_info(summary: ReviewSummary) -> None:
    """Display the LLM provider, model, and budget timing from the completed review."""
    provider = summary.llm_provider or "unknown"
    model = summary.llm_model or "unknown"
    elapsed = summary.semantic_elapsed_seconds
    budget = summary.semantic_budget_seconds
    parts = [f"**Provider:** `{provider}`", f"**Model:** `{model}`"]
    if elapsed or budget:
        parts.append(f"**Elapsed / Budget:** {elapsed:.1f}s / {budget}s")
    st.caption(" · ".join(parts))


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
        use_mock_llm = st.checkbox("Use Mock LLM", value=False)

        # Pre-flight Ollama model availability check (non-blocking display)
        if not use_mock_llm:
            _ollama_ok, _ollama_msg = _check_ollama_model(model.strip() if model else "")
            if _ollama_ok:
                st.caption(_ollama_msg)
            else:
                st.warning(_ollama_msg)

        timeout = st.slider(
            "Ollama Timeout (seconds)",
            30,
            300,
            180,
            help=(
                "Per-request socket timeout. On CPU with a large model, individual "
                "prompts can take 60–180s. The aggregate budget controls total review time."
            ),
        )
        max_files = st.slider(
            "Max Files Budget",
            1,
            20,
            2,
            help=(
                "Maximum changed files per semantic-review batch. "
                "Keep at 1–2 for CPU-only machines with 3B/7B models to avoid timeouts. "
                "Larger values pack more files per request but each request takes longer."
            ),
        )

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

    # --- Model and provider identification ---
    _render_model_info(summary)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Total Findings", summary.total_findings)
    metric_columns[1].metric("Static Findings", summary.static_findings)
    metric_columns[2].metric("Semantic Findings", summary.semantic_findings)
    severity_column = metric_columns[3]
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
    metric_columns[4].metric("Semantic Status", semantic_status)
    reviewed_count = len(summary.reviewed_files) + len(summary.partially_reviewed_files)
    total_scope = (
        reviewed_count
        + len(summary.unreviewed_files)
        + len(summary.skipped_files)
    )
    st.metric(
        "Reviewed Files",
        f"{reviewed_count} of {total_scope}",
    )

    # Budget exhaustion and root failure diagnostics
    if summary.budget_exhausted:
        st.warning(
            f"⏱️ Aggregate review budget of {summary.semantic_budget_seconds}s was exhausted. "
            f"{len(summary.unreviewed_files)} file(s) were not reviewed. "
            "Increase the review budget or reduce the file scope to expand coverage."
        )
    if summary.root_failure_category:
        st.error(
            f"🔴 Root failure: [{summary.root_failure_category}] "
            f"{summary.root_failure_message or 'no details'}"
        )
    if summary.semantic_status in ("partial", "partial_failure", "failed"):
        st.warning(
            "⚠️ Semantic review was **incomplete**. Static findings are retained but "
            "semantic analysis did not cover all files. This is **not** a fully completed review."
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
