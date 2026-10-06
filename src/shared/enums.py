from enum import Enum


class FindingCategory(str, Enum):
    SECURITY = "security"
    STATIC_ANALYSIS = "static_analysis"
    SEMANTIC = "semantic"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
