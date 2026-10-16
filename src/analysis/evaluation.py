"""Evaluation and reproducibility framework for deterministic PR analysis.

Provides standardized comparison of static and security findings against
ground-truth defect datasets, calculating precision, recall, F1, and line localization accuracy.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.shared.models import AnalysisFinding, PRContext
from src.analysis.analyzer import analyze_pr


@dataclass(frozen=True)
class GroundTruthDefect:
    """A ground-truth defect label for benchmarking."""
    file_path: str
    line_number: Optional[int]
    category: str
    rule_id: Optional[str] = None
    description: str = ""


@dataclass
class EvaluationMetric:
    """Evaluation metrics for a test suite or benchmark dataset."""
    total_expected: int
    total_reported: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    exact_localization_matches: int
    localization_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_findings(
    findings: Sequence[AnalysisFinding],
    ground_truth: Sequence[GroundTruthDefect],
    line_tolerance: int = 0,
) -> EvaluationMetric:
    """Evaluate reported findings against ground-truth defects.
    
    Args:
        findings: Findings output by analyze_pr.
        ground_truth: Known true defects.
        line_tolerance: Allowable line difference (0 for exact match).
    
    Returns:
        EvaluationMetric with calculated precision, recall, F1, and localization accuracy.
    """
    total_expected = len(ground_truth)
    total_reported = len(findings)

    if total_expected == 0 and total_reported == 0:
        return EvaluationMetric(
            total_expected=0,
            total_reported=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            exact_localization_matches=0,
            localization_accuracy=1.0,
        )

    matched_gt: set[int] = set()
    matched_findings: set[int] = set()
    exact_line_matches = 0

    for f_idx, finding in enumerate(findings):
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            # Check file match
            if finding.file_path != gt.file_path:
                continue

            # Check category match
            if finding.category.lower() != gt.category.lower():
                continue

            # Check line match
            if finding.line_number is None and gt.line_number is None:
                matched_gt.add(gt_idx)
                matched_findings.add(f_idx)
                exact_line_matches += 1
                break
            elif finding.line_number is not None and gt.line_number is not None:
                diff = abs(finding.line_number - gt.line_number)
                if diff <= line_tolerance:
                    matched_gt.add(gt_idx)
                    matched_findings.add(f_idx)
                    if diff == 0:
                        exact_line_matches += 1
                    break

    tp = len(matched_gt)
    fp = total_reported - tp
    fn = total_expected - tp

    precision = tp / total_reported if total_reported > 0 else (1.0 if total_expected == 0 else 0.0)
    recall = tp / total_expected if total_expected > 0 else (1.0 if total_reported == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    loc_acc = (exact_line_matches / tp) if tp > 0 else 0.0

    return EvaluationMetric(
        total_expected=total_expected,
        total_reported=total_reported,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        exact_localization_matches=exact_line_matches,
        localization_accuracy=round(loc_acc, 4),
    )


def run_benchmark_suite(
    pr_fixtures: Sequence[PRContext],
    ground_truth_map: Dict[int, Sequence[GroundTruthDefect]],
    line_tolerance: int = 0,
) -> Dict[str, Any]:
    """Run automated benchmark suite on PR contexts and compute reproducibility report."""
    results_by_pr: Dict[str, Any] = {}
    all_findings: List[AnalysisFinding] = []
    all_gt: List[GroundTruthDefect] = []

    for pr in pr_fixtures:
        findings = analyze_pr(pr)
        gt = ground_truth_map.get(pr.pr_number, [])
        metric = evaluate_findings(findings, gt, line_tolerance=line_tolerance)
        results_by_pr[f"PR_{pr.pr_number}"] = {
            "title": pr.title,
            "metric": metric.to_dict(),
            "findings_count": len(findings),
            "expected_count": len(gt),
        }
        all_findings.extend(findings)
        all_gt.extend(gt)

    overall_metric = evaluate_findings(all_findings, all_gt, line_tolerance=line_tolerance)

    return {
        "overall": overall_metric.to_dict(),
        "prs": results_by_pr,
    }
