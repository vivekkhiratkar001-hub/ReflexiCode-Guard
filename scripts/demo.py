import argparse
import json
from dataclasses import asdict

from src.github import review_pr
from src.llm import OllamaReviewer
from src.shared.models import PRContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local PR review demo.")
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Enable semantic review through local Ollama.",
    )
    parser.add_argument(
        "--repository-root",
        default=".",
        help="Directory containing optional .reviewrules.",
    )
    args = parser.parse_args()

    context = PRContext(
        repository_owner="demo",
        repository_name="sample",
        pr_number=1,
        title="Add a calculation helper",
        description="Demonstration pull request.",
        issue_context="",
        base_sha="base-sha",
        head_sha="head-sha",
        changed_files=["src/example.py"],
        diff=(
            "diff --git a/src/example.py b/src/example.py\n"
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def calculate(expression):\n"
            "+    return eval(expression)\n"
        ),
    )
    reviewer = OllamaReviewer() if args.ollama else None
    result = review_pr(context, args.repository_root, reviewer)
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
