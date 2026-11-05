from pathlib import Path
from typing import Union


def load_repository_rules(repository_root: Union[str, Path]) -> str:
    """Load optional plain-text review instructions from .reviewrules or docs/requirements.md."""
    root = Path(repository_root)
    parts: list[str] = []
    primary = root / ".reviewrules"
    if primary.is_file():
        parts.append(primary.read_text(encoding="utf-8").strip())
    for candidate in (root / "docs" / "requirements.md", root / "requirements.md"):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text and text not in parts:
                parts.append(text)
    return "\n\n".join(parts)
