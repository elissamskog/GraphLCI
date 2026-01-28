"""
KnowledgeAudit v5 Evaluation Framework
Location: evaluation/__init__.py
Owner: ROOT Instance

Provides pipeline evaluation, intrinsic validation, ground-truth fixtures,
two-stage metrics, and data loaders for SciFact / WikiFactClaims benchmarks.
"""

from evaluation.data_loaders.scifact_loader import SciFactLoader, load_scifact
from evaluation.data_loaders.wikifactclaims_loader import WikiFactClaimsLoader

# v5 evaluation
from evaluation.v5_metrics import V5IntrinsicValidator, V5PipelineSnapshot
from evaluation.v5_evaluator import V5PipelineEvaluator, Feedback
from evaluation.v5_fixtures import (
    V5_ALL_FIXTURES,
    evaluate_v5_ground_truth,
    get_v5_fixture,
)
from evaluation.two_stage_metrics import (
    evaluate_two_stage,
    evaluate_stage_a,
    evaluate_stage_b,
    build_two_stage_report,
)

__all__ = [
    # Data loaders
    "SciFactLoader",
    "load_scifact",
    "WikiFactClaimsLoader",
    # v5
    "V5IntrinsicValidator",
    "V5PipelineSnapshot",
    "V5PipelineEvaluator",
    "Feedback",
    "V5_ALL_FIXTURES",
    "evaluate_v5_ground_truth",
    "get_v5_fixture",
    # Two-stage metrics
    "evaluate_two_stage",
    "evaluate_stage_a",
    "evaluate_stage_b",
    "build_two_stage_report",
]
