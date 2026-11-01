"""New invented case, separate from every frozen case bank and result."""
from copy import deepcopy
import json

MODES = {
    "package_fixed": {"label": "Scoring and charts: fixed answers",
        "description": "Inspect AI scores invented offers with the existing research scorer. No model generates these answers."},
    "package_tiny": {"label": "State editing: tiny CPU model",
        "description": "Steerability edits a tiny randomly initialized model on this computer. It is not trained to negotiate; invalid answers remain unscored."},
}
EXAMPLES = {
    "different_decision": {"title": "Different terms and payoffs", "description": "Fixed offers test whether the scorer and charts preserve changes in both parties' payoffs."},
    "same_decision": {"title": "Same decision", "description": "The edited condition returns the same fixed offer as baseline."},
    "rejected_offer": {"title": "Rejected offer", "description": "The offer is valid, but the supplier rejects its low payoff."},
    "invalid_answer": {"title": "Incomplete answer", "description": "The edited condition omits a term. It must not receive a payoff."},
    "interrupted": {"title": "Interrupted comparison", "description": "A deliberate error after baseline tests preservation and the no-retry rule."},
}
CONDITIONS = ("baseline", "internal", "prompt")
LABELS = {"baseline": "Baseline", "internal": "Internal edit", "prompt": "Prompting"}
ADDITION = "Choose an offer that improves the combined payoff while giving both parties an acceptable payoff."

# Payoffs are invented points, not dollars. Only the buyer's table and the
# supplier's stated priorities are shown in the model-visible prompt.
ISSUES = [
    {"issue_id": "price", "label": "Price per chair", "options": [
        {"option_id": "p60", "label": "$60", "self_payoff": 60, "counterpart_payoff": 0},
        {"option_id": "p80", "label": "$80", "self_payoff": 40, "counterpart_payoff": 20},
        {"option_id": "p100", "label": "$100", "self_payoff": 20, "counterpart_payoff": 40}]},
    {"issue_id": "delivery", "label": "Delivery", "options": [
        {"option_id": "d_week", "label": "This week", "self_payoff": 30, "counterpart_payoff": 0},
        {"option_id": "d_next", "label": "Next week", "self_payoff": 20, "counterpart_payoff": 25},
        {"option_id": "d_month", "label": "Next month", "self_payoff": 0, "counterpart_payoff": 30}]},
    {"issue_id": "warranty", "label": "Warranty", "options": [
        {"option_id": "w1", "label": "One year", "self_payoff": 0, "counterpart_payoff": 20},
        {"option_id": "w2", "label": "Two years", "self_payoff": 25, "counterpart_payoff": 10},
        {"option_id": "w3", "label": "Three years", "self_payoff": 30, "counterpart_payoff": 0}]},
    {"issue_id": "payment", "label": "Payment", "options": [
        {"option_id": "upfront", "label": "Upfront", "self_payoff": 10, "counterpart_payoff": 10},
        {"option_id": "net30", "label": "In 30 days", "self_payoff": 5, "counterpart_payoff": 5},
        {"option_id": "net60", "label": "In 60 days", "self_payoff": 0, "counterpart_payoff": 0}]},
]
BASE = {"price": "p80", "delivery": "d_week", "warranty": "w2", "payment": "net30"}
EDITED = {"price": "p60", "delivery": "d_next", "warranty": "w2", "payment": "upfront"}
PROMPTED = {**BASE, "price": "p100"}
REJECTED = {"price": "p60", "delivery": "d_week", "warranty": "w3", "payment": "net60"}


def authored_case():
    identifier = "local-package-test-chairs"
    return {"schema_version": 1, "output_contract_version": 2,
        "case_id": identifier, "family_id": "local-package-test-only", "variant_id": "base",
        "variant_of": identifier, "variant_kind": "base", "prompt_style": "base",
        "partition": "quality", "domain": "procurement", "role": "buyer",
        "scenario": "A buyer and supplier agree the price, delivery, warranty and payment terms for office chairs. This is an invented software test, not a research observation.",
        "issues": deepcopy(ISSUES), "distributive_issue": "price",
        "integrative_issues": ["delivery", "warranty"], "compatible_issue": "payment",
        "reservation_values": {"self": 30, "counterpart": 30},
        "disagreement_values": {"self": 0, "counterpart": 0},
        "counterpart_rule": {"type": "reservation_rule", "on_tie": "accept"},
        "hard_constraints": [], "counterpart_last_package": deepcopy(BASE),
        "counterpart_stated_priorities": ["Later delivery helps our production schedule.", "We prefer payment upfront."],
        "integrative_opportunity_required": True, "normalization_required": True,
        "sentinel_unrestricted_concession": False,
        "source": {"rights_status": "synthetic_project_authored", "authoring_note": "Invented solely for the October 29 local package integration check; never inserted in a frozen case bank."}}


def fixture_response(example, strength, condition):
    if condition == "baseline" or (condition == "internal" and strength == 0):
        terms = BASE
    elif condition == "prompt":
        terms = PROMPTED
    elif example == "interrupted":
        raise RuntimeError("Deliberate local fixture interruption")
    elif example == "same_decision":
        terms = BASE
    elif example == "invalid_answer":
        terms = {"price": "p60", "delivery": "d_next"}
    elif example == "rejected_offer" or strength in (-1, 2):
        terms = REJECTED
    else:
        terms = EDITED
    return json.dumps({"decision": "offer", "package": terms}, sort_keys=True, separators=(",", ":"))


def display_case():
    return {"id": "local-package-test-chairs", "title": "Chair supply agreement",
        "description": "Four terms determine the buyer's and supplier's payoffs: price, delivery, warranty and payment. The values below are invented points for a software test.",
        "units": "points", "issues": deepcopy(ISSUES), "outside_payoffs": {"buyer": 0, "supplier": 0},
        "acceptance_rule": "The scripted supplier accepts an offer at 30 points or more. If it rejects, both parties receive 0 points. Invalid responses have no payoff score."}


def empty_result(mode, example, strength):
    if mode not in MODES or example not in EXAMPLES or type(strength) is not int or strength not in (-1, 0, 1, 2):
        raise ValueError("Unsupported local test")
    tiny = mode == "package_tiny"
    result = {"schema": "local-package-integration/1", "mode": mode, "simulation": True,
        "scientific_evidence": False, "research_model_execution": False, "model_execution": False,
        "model_execution_planned": tiny, "provider": "none", "paid_runs": 0, "units": "points",
        "worker": "Steerability + Inspect AI, local CPU only",
        "notice": MODES[mode]["description"], "case": display_case(), "example": {"id": example, **EXAMPLES[example]},
        "strength": strength, "conditions": [], "execution": {"generation_calls": 0, "network_attempts": [], "device": "cpu"}}
    for condition in CONDITIONS:
        result["conditions"].append({"condition": condition, "label": LABELS[condition], "status": "not_run",
            "reason": "Not run yet.", "raw_response": None, "offer": None, "scores": None,
            "instructions": None, "added_instruction": ADDITION if condition == "prompt" else None,
            "edit": {"actual_state_edit": False, "requested_setting": strength} if condition == "internal" else None})
    return result


def catalog():
    return {"modes": [{"id": key, **value} for key, value in MODES.items()],
        "case": display_case(), "examples": [{"id": key, **value} for key, value in EXAMPLES.items()],
        "base_instructions": "The exact model-visible instructions appear in the result after the test. They include the buyer's payoff table and the supplier's stated priorities, not its private payoff table.",
        "prompt_addition": ADDITION, "strengths": [-1, 0, 1, 2]}
