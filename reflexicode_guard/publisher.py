"""Day 5-6 (4-5 Oct): GitHub review-comment publisher."""
from __future__ import annotations

import re
from .github_client import GitHubClient, GitHubError
from .models import Finding, PRContext

MARK = "<!-- reflexicode-guard:{fp} -->"
MARK_RE = re.compile(r"<!-- reflexicode-guard:([0-9a-f]{12}) -->")
ICON = {"critical": "🛑", "high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}


def format_comment(f: Finding) -> str:
    rule = f" · `{f.rule_id}`" if f.rule_id else ""
    return (f"{ICON[f.severity]} **{f.severity.upper()}** · {f.category} · _{f.source}_{rule}\n\n"
            f"{f.message}\n\n**Evidence**\n```\n{f.evidence}\n```\n{MARK.format(fp=f.fingerprint)}")


def _unreviewed_note(stats: dict) -> str:
    n = stats.get("unreviewable_files", 0)
    return f"\n\n⚠️ {n} file(s) could not be reviewed (GitHub provided no diff, e.g. very large or binary)." if n else ""


def format_summary(findings: list[Finding], stats: dict) -> str:
    if not findings:
        return "### ReflexiCode-Guard\nNo issues found in the changed lines. ✅" + _unreviewed_note(stats)
    by_sev = {}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    parts = ", ".join(f"{n} {s}" for s, n in sorted(by_sev.items(), key=lambda kv: -["info","low","medium","high","critical"].index(kv[0])))
    extra = f"\n\n_{stats['truncated']} lower-priority finding(s) omitted._" if stats.get("truncated") else ""
    return f"### ReflexiCode-Guard\nFound **{len(findings)}** issue(s): {parts}.{extra}" + _unreviewed_note(stats)


def build_review_payload(findings: list[Finding], stats: dict) -> dict:
    return {"body": format_summary(findings, stats),
            "comments": [{"path": f.file, "line": f.line, "side": "RIGHT", "body": format_comment(f)}
                         for f in findings]}


def publish(client: GitHubClient, ctx: PRContext, findings: list[Finding], stats: dict,
            dry_run: bool = False) -> dict:
    """Post one PR review. Skips findings already posted (by fingerprint marker)."""
    result = {"posted": 0, "skipped_existing": 0, "fallback": False, "dry_run": dry_run}
    existing = set()
    if not dry_run:
        for c in client.list_review_comments(ctx.repo, ctx.number):
            existing.update(MARK_RE.findall(c.get("body", "")))
    fresh = [f for f in findings if f.fingerprint not in existing]
    result["skipped_existing"] = len(findings) - len(fresh)
    if not fresh and findings:
        return result                          # nothing new -> stay quiet on re-runs
    payload = build_review_payload(fresh, stats)
    result["payload"] = payload
    if dry_run:
        result["posted"] = len(fresh)
        return result
    try:
        client.create_review(ctx.repo, ctx.number, ctx.head_sha, payload["body"], payload["comments"])
        result["posted"] = len(fresh)
    except GitHubError as e:
        if e.status != 422:
            raise
        # A line target was rejected (e.g. force-push moved the diff): degrade to one issue comment.
        lines = [f"- `{f.file}:{f.line}` **{f.severity}** {f.message}" for f in fresh]
        client.create_issue_comment(ctx.repo, ctx.number, payload["body"] + "\n\n" + "\n".join(lines))
        result.update(posted=len(fresh), fallback=True)
    return result
