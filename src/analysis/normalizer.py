"""Finding normalization and deduplication for static and security analysis."""

from typing import Dict, Iterable, List, Optional, Tuple

from src.shared.enums import FindingCategory, Severity
from src.shared.models import AnalysisFinding
from src.analysis.rules.base import RuleFinding


_SEVERITY_MAP = {
    "critical": Severity.CRITICAL.value,
    "high": Severity.HIGH.value,
    "medium": Severity.MEDIUM.value,
    "low": Severity.LOW.value,
}

_CATEGORY_MAP = {
    "security": FindingCategory.SECURITY.value,
    "static_analysis": FindingCategory.STATIC_ANALYSIS.value,
    "semantic": FindingCategory.SEMANTIC.value,
}


def normalize_finding(
    file_path: str,
    finding: RuleFinding,
) -> AnalysisFinding:
    """Normalize a RuleFinding into the shared AnalysisFinding contract."""
    clean_path = file_path.strip().replace("\\", "/")
    norm_category = _CATEGORY_MAP.get(finding.category.lower(), FindingCategory.STATIC_ANALYSIS.value)
    norm_severity = _SEVERITY_MAP.get(finding.severity.lower(), Severity.MEDIUM.value)

    return AnalysisFinding(
        file_path=clean_path,
        line_number=finding.line_number,
        category=norm_category,
        severity=norm_severity,
        message=finding.message.strip(),
        evidence=(finding.evidence or "").strip(),
        suggestion=finding.suggestion.strip(),
    )


def deduplicate_and_sort_findings(
    file_path: str,
    raw_findings: Iterable[RuleFinding],
) -> List[AnalysisFinding]:
    """Deduplicate findings by identity and return deterministically ordered list.

    Identical defects on the same file and line with the same message are deduplicated.
    Distinct defects occurring on the same line are preserved.
    """
    findings_by_identity: Dict[Tuple[str, Optional[int], str, str], AnalysisFinding] = {}

    for rf in raw_findings:
        normalized = normalize_finding(file_path, rf)
        identity = (
            normalized.file_path,
            normalized.line_number,
            normalized.category,
            normalized.message,
        )
        if identity not in findings_by_identity:
            findings_by_identity[identity] = normalized

    return sorted(
        findings_by_identity.values(),
        key=lambda item: (
            item.file_path,
            item.line_number if item.line_number is not None else -1,
            item.category,
            item.message,
        ),
    )
