"""Shared saved-response interfaces. No model loader, fit, selection or launch."""
from dataclasses import dataclass
from typing import Protocol
import json
from .reuse import legacy
from .scoring import score_completed


class OutcomeScorer(Protocol):
    name: str
    outcomes: tuple[str, ...]
    def score(self, case: dict, raw: str) -> dict: ...


@dataclass(frozen=True)
class Study:
    name: str
    cases: tuple[dict, ...]
    conditions: tuple[str, ...]
    fixture_only: bool


class NegotiationScorer:
    name = "negotiation"
    outcomes = ("buyer", "supplier", "combined", "agreement_adjusted_combined")
    def score(self, case, raw):
        score = score_completed(legacy.scorer(), case, raw)
        return {"valid": score["valid"], "errors": score["errors"], "decision": score["decision"],
            "outcomes": dict(zip(self.outcomes, [score[k] for k in
                ("proposed_self_payoff", "proposed_counterpart_payoff", "proposed_joint_gain", "agreement_adjusted_joint_gain")])),
            "detail": score}


class AllocationScorer:
    """Authored synthetic second task; not a new empirical study."""
    name = "synthetic-allocation"
    outcomes = ("team_a", "team_b", "absolute_gap")
    def score(self, case, raw):
        invalid = {"valid": False, "errors": ["ALLOCATION"], "decision": None,
                   "outcomes": {name: None for name in self.outcomes}}
        try:
            value = json.loads(raw, object_pairs_hook=legacy._unique_object)
        except (ValueError, TypeError):
            return invalid
        if (not isinstance(value, dict) or set(value) != {"team_a", "team_b"}
                or any(type(v) is not int or v < 0 for v in value.values())
                or sum(value.values()) != case["budget"]):
            return invalid
        return {"valid": True, "errors": [], "decision": "allocate",
            "outcomes": {**value, "absolute_gap": abs(value["team_a"] - value["team_b"])}}


def evaluate_saved(study: Study, scorer: OutcomeScorer, records):
    """One attempt per case/condition, all declared cases retained, no auto-selection."""
    if study.fixture_only is not True:
        raise legacy.AnalysisError("This reusable interface is qualified for synthetic software checks only")
    cases = {c["case_id"]: c for c in study.cases}
    if not cases or not study.conditions or len(cases) != len(study.cases) or len(set(study.conditions)) != len(study.conditions):
        raise legacy.AnalysisError("Cases and conditions must be nonempty and unique")
    by_pair = {}
    for row in records:
        key = row["case_id"], row["condition"]
        if key in by_pair or key[0] not in cases or key[1] not in study.conditions:
            raise legacy.AnalysisError("Duplicate or undeclared reusable-workflow attempt")
        by_pair[key] = row
    rows = []
    for case in study.cases:
        for condition in study.conditions:
            row = by_pair.get((case["case_id"], condition))
            status = row["execution_status"] if row else "INDETERMINATE"
            if status not in {"COMPLETED", "TECHNICAL_FAILURE", "INDETERMINATE"}:
                raise legacy.AnalysisError("Unknown execution status")
            raw = row.get("raw_response") if row else None
            score = scorer.score(case, raw) if status == "COMPLETED" else {
                "valid": False, "errors": [status], "decision": None,
                "outcomes": {name: None for name in scorer.outcomes}}
            rows.append({"case_id": case["case_id"], "condition": condition, "execution_status": status,
                "raw_response": raw, "source": row.get("source") if row else None, **score})
    return {"artifact_type": "REUSABLE_SYNTHETIC_WORKFLOW_CHECK", "name": study.name,
        "fixture_only": True, "evidence_label": "SYNTHETIC_TEST_ONLY", "model_execution": False,
        "provider_calls": 0, "outcome_names": list(scorer.outcomes), "rows": rows,
        "selection_performed": False, "empirical_evidence": False}
