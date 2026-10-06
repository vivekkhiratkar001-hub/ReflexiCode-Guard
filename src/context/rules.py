from pathlib import Path
from typing import Union


def load_repository_rules(repository_root: Union[str, Path]) -> str:
    """Load optional plain-text review instructions from .reviewrules."""
    rules_file = Path(repository_root) / ".reviewrules"
    if not rules_file.is_file():
        return ""
    return rules_file.read_text(encoding="utf-8")
