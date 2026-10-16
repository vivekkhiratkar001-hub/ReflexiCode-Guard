"""Evaluation benchmark script for Member 2 static and security analysis."""

import json
from pathlib import Path

from src.analysis.evaluation import run_benchmark_suite
from tests.fixtures import (
    get_changed_vs_unchanged_pr,
    get_clean_code_pr,
    get_complexity_violation_pr,
    get_deterministic_defect_pr,
    get_duplicate_defect_pr,
    get_file_lifecycle_pr,
    get_ground_truth_map,
    get_security_defect_pr,
)


def main() -> None:
    fixtures = [
        get_clean_code_pr(),
        get_deterministic_defect_pr(),
        get_security_defect_pr(),
        get_complexity_violation_pr(),
        get_changed_vs_unchanged_pr(),
        get_duplicate_defect_pr(),
        get_file_lifecycle_pr(),
    ]
    gt_map = get_ground_truth_map()

    print("Running Member 2 Evaluation Benchmark Suite...")
    report = run_benchmark_suite(fixtures, gt_map, line_tolerance=0)

    overall = report["overall"]
    print("\n--- Overall Benchmark Results ---")
    print(f"Total Expected Defects:     {overall['total_expected']}")
    print(f"Total Reported Findings:    {overall['total_reported']}")
    print(f"True Positives:             {overall['true_positives']}")
    print(f"False Positives:            {overall['false_positives']}")
    print(f"False Negatives:            {overall['false_negatives']}")
    print(f"Precision:                  {overall['precision']:.4f}")
    print(f"Recall:                     {overall['recall']:.4f}")
    print(f"F1 Score:                   {overall['f1_score']:.4f}")
    print(f"Exact Localization Matches: {overall['exact_localization_matches']}")
    print(f"Localization Accuracy:      {overall['localization_accuracy'] * 100:.1f}%")

    output_path = Path("benchmark_results.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nMachine-readable evaluation artifact saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
