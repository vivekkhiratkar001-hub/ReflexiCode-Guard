"""Deterministic evaluation helpers for review findings.

This module is intentionally separate from the production review pipeline and is
used only after a model response has already been parsed and validated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .schemas import ReviewFinding


class GroundTruthFinding(BaseModel):
    """A manually defined issue that should be detected by the review model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    file: StrictStr = Field(min_length=1)
    line: StrictInt = Field(gt=0)
    category: StrictStr = Field(min_length=1)
    severity: StrictStr = Field(min_length=1)
    message: StrictStr = Field(min_length=1)


class EvaluationCase(BaseModel):
    """A small review dataset case with expected findings."""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: StrictStr = Field(min_length=1)
    description: StrictStr = ""
    changed_code: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    changed_lines: dict[StrictStr, list[int]] = Field(default_factory=dict)
    repository_rules: list[dict[str, Any]] = Field(default_factory=list)
    static_analysis_findings: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth_findings: list[GroundTruthFinding] = Field(default_factory=list)


class CaseEvaluationResult(BaseModel):
    """TP/FP/FN metrics for one evaluation case."""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: StrictStr
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class EvaluationResult(BaseModel):
    """Aggregate metrics for a set of review cases."""

    model_config = ConfigDict(extra="forbid", strict=True)

    cases: list[CaseEvaluationResult] = Field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./").casefold()


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


class ReviewEvaluator:
    """Evaluate predicted findings against ground-truth issues."""

    def __init__(self, line_tolerance: int = 1) -> None:
        self.line_tolerance = max(0, int(line_tolerance))

    def _files_match(self, truth_file: str, predicted_file: str) -> bool:
        return _normalize_path(truth_file) == _normalize_path(predicted_file)

    def _categories_match(self, truth_category: str, predicted_category: str) -> bool:
        return truth_category.casefold() == predicted_category.casefold()

    def _line_matches(self, truth_line: int, predicted_line: int) -> bool:
        return abs(truth_line - predicted_line) <= self.line_tolerance

    def matches(self, truth: GroundTruthFinding, prediction: ReviewFinding) -> bool:
        """Return True when the prediction and ground truth describe the same issue."""
        return (
            self._files_match(truth.file, prediction.file)
            and self._categories_match(truth.category, prediction.category)
            and self._line_matches(truth.line, prediction.line)
        )

    def _best_match_pairs(
        self,
        ground_truth_findings: Sequence[GroundTruthFinding],
        predicted_findings: Sequence[ReviewFinding],
    ) -> list[tuple[int, int]]:
        """Compute the maximum-cardinality matching with deterministic ordering."""
        truth_count = len(ground_truth_findings)
        prediction_count = len(predicted_findings)
        memo: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = {}

        def search(truth_index: int, used_predictions: tuple[int, ...]) -> list[tuple[int, int]]:
            if truth_index >= truth_count:
                return []
            key = (truth_index, used_predictions)
            if key in memo:
                return memo[key]

            best: list[tuple[int, int]] = []
            for prediction_index in range(prediction_count):
                if prediction_index in used_predictions:
                    continue
                if not self.matches(ground_truth_findings[truth_index], predicted_findings[prediction_index]):
                    continue
                match = [(truth_index, prediction_index)]
                future = search(truth_index + 1, tuple(sorted((*used_predictions, prediction_index))))
                candidate = match + future
                if len(candidate) > len(best):
                    best = candidate

            skip = search(truth_index + 1, used_predictions)
            if len(skip) > len(best):
                best = skip

            memo[key] = best
            return best

        return search(0, ())

    def evaluate_case(
        self,
        case_id: str,
        ground_truth_findings: Sequence[GroundTruthFinding],
        predicted_findings: Sequence[ReviewFinding],
    ) -> CaseEvaluationResult:
        """Compute TP/FP/FN for one case and return safe metric values."""
        matches = self._best_match_pairs(ground_truth_findings, predicted_findings)
        tp = len(matches)
        fp = len(predicted_findings) - tp
        fn = len(ground_truth_findings) - tp

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        return CaseEvaluationResult(
            case_id=case_id,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def evaluate_cases(
        self,
        cases: Sequence[EvaluationCase],
        predictions_by_case: Mapping[str, Sequence[ReviewFinding]],
    ) -> EvaluationResult:
        """Evaluate multiple cases and aggregate the overall metrics."""
        case_results: list[CaseEvaluationResult] = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for case in cases:
            predicted = list(predictions_by_case.get(case.case_id, ()))
            result = self.evaluate_case(
                case.case_id,
                case.ground_truth_findings,
                predicted,
            )
            case_results.append(result)
            total_tp += result.tp
            total_fp += result.fp
            total_fn += result.fn

        overall_precision = _safe_divide(total_tp, total_tp + total_fp)
        overall_recall = _safe_divide(total_tp, total_tp + total_fn)
        overall_f1 = _safe_divide(2 * overall_precision * overall_recall, overall_precision + overall_recall)

        return EvaluationResult(
            cases=case_results,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            precision=overall_precision,
            recall=overall_recall,
            f1=overall_f1,
        )


def evaluate_predictions(
    ground_truth_findings: Sequence[GroundTruthFinding],
    predicted_findings: Sequence[ReviewFinding],
    line_tolerance: int = 1,
) -> CaseEvaluationResult:
    """Convenience wrapper for evaluating one ground-truth/prediction pair."""
    evaluator = ReviewEvaluator(line_tolerance=line_tolerance)
    return evaluator.evaluate_case("case", ground_truth_findings, predicted_findings)
