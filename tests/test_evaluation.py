import json
from pathlib import Path

from src.llm.evaluation import EvaluationCase, GroundTruthFinding, ReviewEvaluator
from src.llm.schemas import ReviewFinding


FIXTURE_CASES = [
    EvaluationCase.model_validate(case)
    for case in json.loads((Path(__file__).parent / "fixtures" / "evaluation_cases.json").read_text(encoding="utf-8"))
]


def test_perfect_prediction() -> None:
    evaluator = ReviewEvaluator()
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk")]
    predicted = [ReviewFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk", evidence="query = ...")]

    result = evaluator.evaluate_case("case-1", truth, predicted)

    assert result.tp == 1
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_completely_missed_issue() -> None:
    evaluator = ReviewEvaluator()
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk")]
    predicted: list[ReviewFinding] = []

    result = evaluator.evaluate_case("case-2", truth, predicted)

    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 1
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_completely_incorrect_prediction() -> None:
    evaluator = ReviewEvaluator()
    truth: list[GroundTruthFinding] = []
    predicted = [ReviewFinding(file="src/app.py", line=50, category="Bug", severity="Low", message="Issue found", evidence="something unrelated")]

    result = evaluator.evaluate_case("case-3", truth, predicted)

    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_one_correct_and_one_incorrect_prediction() -> None:
    evaluator = ReviewEvaluator()
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection")]
    predicted = [
        ReviewFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk", evidence="query = ..."),
        ReviewFinding(file="src/other.py", line=2, category="Bug", severity="Low", message="Unrelated bug", evidence="other code"),
    ]

    result = evaluator.evaluate_case("case-4", truth, predicted)

    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 0
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.f1 == 0.6666666666666666


def test_one_missed_and_one_correct() -> None:
    evaluator = ReviewEvaluator()
    truth = [
        GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection"),
        GroundTruthFinding(file="src/app.py", line=20, category="Performance", severity="Medium", message="Slow loop"),
    ]
    predicted = [
        ReviewFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk", evidence="query = ..."),
    ]

    result = evaluator.evaluate_case("case-5", truth, predicted)

    assert result.tp == 1
    assert result.fp == 0
    assert result.fn == 1
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.f1 == 0.6666666666666666


def test_line_tolerance_allows_small_difference() -> None:
    evaluator = ReviewEvaluator(line_tolerance=1)
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection")]
    predicted = [ReviewFinding(file="src/app.py", line=13, category="Security", severity="High", message="SQL injection risk", evidence="query = ...")]

    result = evaluator.evaluate_case("case-6", truth, predicted)

    assert result.tp == 1
    assert result.fp == 0
    assert result.fn == 0


def test_overlap_resolution_maximizes_all_valid_matches() -> None:
    evaluator = ReviewEvaluator(line_tolerance=1)
    truth = [
        GroundTruthFinding(file="src/app.py", line=10, category="Bug", severity="High", message="Null dereference"),
        GroundTruthFinding(file="src/app.py", line=12, category="Bug", severity="High", message="Boundary check issue"),
    ]
    predicted = [
        ReviewFinding(file="src/app.py", line=11, category="Bug", severity="High", message="Possible null dereference", evidence="if x is None"),
        ReviewFinding(file="src/app.py", line=10, category="Bug", severity="High", message="Boundary check issue", evidence="if idx < 0"),
    ]

    result = evaluator.evaluate_case("case-6b", truth, predicted)

    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0


def test_large_line_difference_does_not_match() -> None:
    evaluator = ReviewEvaluator(line_tolerance=1)
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection")]
    predicted = [ReviewFinding(file="src/app.py", line=50, category="Security", severity="High", message="SQL injection risk", evidence="query = ...")]

    result = evaluator.evaluate_case("case-7", truth, predicted)

    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1


def test_different_category_does_not_match() -> None:
    evaluator = ReviewEvaluator()
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection")]
    predicted = [ReviewFinding(file="src/app.py", line=12, category="Bug", severity="High", message="SQL injection risk", evidence="query = ...")]

    result = evaluator.evaluate_case("case-8", truth, predicted)

    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1


def test_multiple_ground_truth_findings_are_not_double_counted() -> None:
    evaluator = ReviewEvaluator()
    truth = [
        GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection"),
        GroundTruthFinding(file="src/app.py", line=20, category="Performance", severity="Medium", message="Slow loop"),
    ]
    predicted = [
        ReviewFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk", evidence="query = ..."),
        ReviewFinding(file="src/app.py", line=20, category="Performance", severity="Medium", message="Slow loop risk", evidence="for x in ..."),
        ReviewFinding(file="src/app.py", line=20, category="Performance", severity="Medium", message="Duplicate performance issue", evidence="for x in ..."),
    ]

    result = evaluator.evaluate_case("case-9", truth, predicted)

    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 0


def test_empty_prediction_list_is_safe() -> None:
    evaluator = ReviewEvaluator()
    truth = [GroundTruthFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection")]

    result = evaluator.evaluate_case("case-10", truth, [])

    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 1


def test_empty_ground_truth_list_is_safe() -> None:
    evaluator = ReviewEvaluator()
    predicted = [ReviewFinding(file="src/app.py", line=12, category="Security", severity="High", message="SQL injection risk", evidence="query = ...")]

    result = evaluator.evaluate_case("case-11", [], predicted)

    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 0


def test_fixture_dataset_evaluates_realistic_cases() -> None:
    evaluator = ReviewEvaluator()
    predictions_by_case = {
        "sql-injection-user-service": [
            ReviewFinding(
                file="src/services/user_service.py",
                line=2,
                category="Security",
                severity="High",
                message="Direct SQL interpolation creates an injection risk.",
                evidence="query = f\"SELECT * FROM users WHERE id = {user_id}\"",
            )
        ],
        "repository-layer-violation": [
            ReviewFinding(
                file="src/services/order_service.py",
                line=2,
                category="Architecture",
                severity="Medium",
                message="Database access should go through repository abstraction.",
                evidence="return db.execute(f\"SELECT * FROM orders WHERE id = {order_id}\")",
            )
        ],
        "performance-loop": [
            ReviewFinding(
                file="src/services/report_service.py",
                line=3,
                category="Performance",
                severity="Medium",
                message="Loop performs per-user database queries, creating N+1 behavior.",
                evidence="rows.append(db.execute(f\"SELECT * FROM profiles WHERE id = {user.id}\"))",
            )
        ],
        "clean-review": [],
        "duplicate-risk": [
            ReviewFinding(
                file="src/services/inventory_service.py",
                line=2,
                category="Security",
                severity="High",
                message="Dynamic SQL in the service layer allows SQL injection.",
                evidence="return db.execute(f\"SELECT * FROM inventory WHERE id = {item_id}\")",
            )
        ],
    }

    result = evaluator.evaluate_cases(FIXTURE_CASES, predictions_by_case)

    assert result.total_tp == 4
    assert result.total_fp == 0
    assert result.total_fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
