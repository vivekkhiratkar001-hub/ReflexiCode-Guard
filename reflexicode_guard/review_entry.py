"""Day 2-4 (1-3 Oct): review entry point used by GitHub Actions and local runs.

  CI:      python -m reflexicode_guard.review_entry
  Offline: python -m reflexicode_guard.review_entry --offline tests/fixtures --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .diff_utils import split_diff_by_file
from .events import load_event, parse_pr_event, should_review, extract_linked_issues
from .github_client import API, GitHubClient
from .models import ChangedFile, PRContext
from .config import load_analyzers, load_config
from .pipeline import run_pipeline
from .publisher import publish

log = logging.getLogger("reflexicode_guard")


def context_from_fixture(folder: str) -> PRContext:
    """Offline mode: build PRContext from pr_event.json + sample.diff (no network)."""
    ev = parse_pr_event(load_event(os.path.join(folder, "pr_event.json")))
    pr = load_event(os.path.join(folder, "pr_event.json"))["pull_request"]
    with open(os.path.join(folder, "sample.diff"), encoding="utf-8") as fh:
        patches = split_diff_by_file(fh.read())
    return PRContext(repo=ev.repo, number=ev.number, title=pr["title"], body=pr.get("body") or "",
                     author=pr["user"]["login"], base_ref=ev.base_ref, head_ref=ev.head_ref,
                     base_sha=pr["base"]["sha"], head_sha=ev.head_sha, draft=ev.draft,
                     files=[ChangedFile(p, "modified", patch) for p, patch in patches.items()],
                     linked_issues=extract_linked_issues(pr.get("body")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ReflexiCode-Guard PR review entry point")
    ap.add_argument("--event", help="path to GitHub event JSON (default: $GITHUB_EVENT_PATH)")
    ap.add_argument("--offline", metavar="DIR", help="fixture folder (pr_event.json + sample.diff); no network")
    ap.add_argument("--pr", type=int, help="PR number (manual/workflow_dispatch runs: no event file needed)")
    ap.add_argument("--repo", help="owner/name (default: $GITHUB_REPOSITORY)")
    ap.add_argument("--dry-run", action="store_true", help="do not post to GitHub; print payload")
    ap.add_argument("--output", default="findings.json", help="where to write the findings artifact")
    ap.add_argument("--force", action="store_true", help="skip the branch-strategy gate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # GitHub Actions sets GITHUB_API_URL automatically (also correct for GitHub Enterprise).
    client = GitHubClient(os.environ.get("GITHUB_TOKEN"), api_url=os.environ.get("GITHUB_API_URL", API))
    if args.offline:
        ctx = context_from_fixture(args.offline)
        ev = parse_pr_event(load_event(os.path.join(args.offline, "pr_event.json")))
        args.dry_run = True
    elif args.pr:
        repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
        if not repo:
            ap.error("--pr needs --repo or $GITHUB_REPOSITORY")
        ev = parse_pr_event({"action": "synchronize", "repository": {"full_name": repo},
                             "pull_request": client.get_pull(repo, args.pr)})
        ctx = None
    else:
        ev = parse_pr_event(load_event(args.event))
        ctx = None

    cfg = load_config()
    ok, reason = should_review(ev, tuple(cfg.protected_bases))
    if not ok and not args.force:
        log.info("Skipping review: %s", reason)
        return 0
    log.info("Reviewing %s#%s @ %s (%s)", ev.repo, ev.number, ev.head_sha[:7], reason)
    if ev.from_fork and not args.dry_run:
        log.warning("Fork PR: forcing dry-run (GITHUB_TOKEN is read-only for forks)")
        args.dry_run = True

    ctx = ctx or client.build_context(ev.repo, ev.number)
    findings, stats = run_pipeline(ctx, load_analyzers(cfg), max_findings=cfg.max_findings)
    result = publish(client, ctx, findings, stats, dry_run=args.dry_run)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"pr": f"{ctx.repo}#{ctx.number}", "head_sha": ctx.head_sha, "stats": stats,
                   "publish": {k: v for k, v in result.items() if k != "payload"},
                   "findings": [f.to_dict() for f in findings]}, fh, indent=2)
    log.info("stats=%s publish=%s", stats, {k: v for k, v in result.items() if k != "payload"})
    if args.dry_run and "payload" in result:
        print(json.dumps(result["payload"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
