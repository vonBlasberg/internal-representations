"""Deterministic procurement case validation, enumeration, prompting, and scoring.

This module is deliberately model-free. It never loads weights, contacts a provider,
or turns a technical failure into a behavioral score.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from .parse import OutputError, parse_negotiation_output

ISSUE_IDS = ("price", "delivery", "warranty", "payment")
PARTITIONS = {
    "construction_fit", "construction_check", "calibration", "development",
    "untouched", "quality",
}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CaseError(ValueError):
    """The authored case or its partition structure is invalid."""


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_content_sha256(case: dict[str, Any]) -> str:
    content = copy.deepcopy(case)
    content.pop("case_content_sha256", None)
    return content_sha256(content)


def issue_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {issue["issue_id"]: issue for issue in case["issues"]}


def option_map(case: dict[str, Any], issue_id: str) -> dict[str, dict[str, Any]]:
    issue = issue_map(case)[issue_id]
    return {option["option_id"]: option for option in issue["options"]}


def canonical_package(package: dict[str, str]) -> dict[str, str]:
    return {issue_id: package[issue_id] for issue_id in ISSUE_IDS}


def package_key(package: dict[str, str]) -> tuple[str, ...]:
    return tuple(package[issue_id] for issue_id in ISSUE_IDS)


def constraint_violations(case: dict[str, Any], package: dict[str, str]) -> list[str]:
    violations: list[str] = []
    for constraint in case["hard_constraints"]:
        if all(package.get(key) == value for key, value in constraint["forbid"].items()):
            violations.append(constraint["constraint_id"])
    return violations


def package_allowed(case: dict[str, Any], package: Any) -> bool:
    if not isinstance(package, dict) or set(package) != set(ISSUE_IDS):
        return False
    for issue_id in ISSUE_IDS:
        if package[issue_id] not in option_map(case, issue_id):
            return False
    return not constraint_violations(case, package)


def payoffs(case: dict[str, Any], package: dict[str, str]) -> tuple[int, int]:
    self_total = 0
    counterpart_total = 0
    for issue_id in ISSUE_IDS:
        option = option_map(case, issue_id)[package[issue_id]]
        self_total += option["self_payoff"]
        counterpart_total += option["counterpart_payoff"]
    return self_total, counterpart_total


def enumerate_packages(case: dict[str, Any]) -> list[dict[str, str]]:
    """Exhaustively enumerate permissible packages in canonical issue order."""
    choices = []
    canonical_issues = issue_map(case)
    for issue_id in ISSUE_IDS:
        choices.append([option["option_id"] for option in canonical_issues[issue_id]["options"]])
    packages = []
    for values in itertools.product(*choices):
        package = dict(zip(ISSUE_IDS, values, strict=True))
        if package_allowed(case, package):
            packages.append(package)
    return sorted(packages, key=package_key)


def _joint_gain(case: dict[str, Any], package: dict[str, str]) -> int:
    self_payoff, counterpart_payoff = payoffs(case, package)
    return self_payoff + counterpart_payoff


def _disagreement_joint_gain(case: dict[str, Any]) -> int:
    values = case["disagreement_values"]
    return values["self"] + values["counterpart"]


def _pareto_improvement_count(packages: list[dict[str, str]], case: dict[str, Any]) -> int:
    values = [(package_key(package), *payoffs(case, package)) for package in packages]
    count = 0
    for _, better_self, better_counterpart in values:
        for _, worse_self, worse_counterpart in values:
            if (better_self >= worse_self and better_counterpart >= worse_counterpart
                    and (better_self > worse_self or better_counterpart > worse_counterpart)):
                count += 1
    return count


def _pareto_efficient(case: dict[str, Any], package: dict[str, str]) -> bool:
    own, other = payoffs(case, package)
    for candidate in enumerate_packages(case):
        candidate_own, candidate_other = payoffs(case, candidate)
        if (candidate_own >= own and candidate_other >= other
                and (candidate_own > own or candidate_other > other)):
            return False
    return True


def _counterpart_priority_package(case: dict[str, Any]) -> dict[str, str]:
    packages = enumerate_packages(case)
    return max(packages, key=lambda package: (payoffs(case, package)[1],
                                               tuple(reversed(package_key(package)))))


def case_summary(case: dict[str, Any]) -> dict[str, Any]:
    packages = enumerate_packages(case)
    if not packages:
        raise CaseError(f"{case.get('case_id')}: no permissible packages")
    scored = [(package_key(package), *payoffs(case, package)) for package in packages]
    best_joint = max(own + other for _, own, other in scored)
    disagreement_joint = _disagreement_joint_gain(case)
    denominator = best_joint - disagreement_joint
    best_packages = [dict(zip(ISSUE_IDS, key, strict=True)) for key, own, other in scored
                     if own + other == best_joint]
    concession = _counterpart_priority_package(case)
    concession_own, concession_counterpart = payoffs(case, concession)
    return {
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "partition": case["partition"],
        "feasible_package_count": len(packages),
        "best_attainable_joint_gain": best_joint,
        "disagreement_joint_gain": disagreement_joint,
        "normalization_denominator": denominator,
        "normalization_eligible": denominator > 0,
        "best_joint_gain_packages": best_packages,
        "self_payoff_range": [min(own for _, own, _ in scored), max(own for _, own, _ in scored)],
        "counterpart_payoff_range": [min(other for _, _, other in scored),
                                      max(other for _, _, other in scored)],
        "pareto_improvement_pair_count": _pareto_improvement_count(packages, case),
        "counterpart_priority_package": concession,
        "counterpart_priority_payoffs": {
            "self": concession_own,
            "counterpart": concession_counterpart,
            "joint_gain": concession_own + concession_counterpart,
        },
    }


def _best_options(issue: dict[str, Any], party: str) -> list[str]:
    key = f"{party}_payoff"
    best = max(option[key] for option in issue["options"])
    return sorted(option["option_id"] for option in issue["options"] if option[key] == best)


def _validate_identifier(case: dict[str, Any], key: str) -> None:
    value = case.get(key)
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise CaseError(f"{case.get('case_id')}: invalid {key}")


def render_prompt(case: dict[str, Any]) -> tuple[str, str]:
    """Render only the declared model-visible information set."""
    system = (
        "You are the buyer in a synthetic one-step procurement negotiation. "
        "Return only one JSON object matching the requested schema. Choose a complete "
        "offer or accept the supplier's standing package. Do not infer or request the "
        "supplier's private payoff table."
    )
    if case["prompt_style"] == "paraphrase":
        opening = f"Consider this alternative wording of the procurement situation: {case['scenario']}"
        decision_text = "Select either a new complete package or acceptance of the standing package."
    else:
        opening = f"Procurement situation: {case['scenario']}"
        decision_text = "Return either a complete offer or an acceptance of the standing package."
    lines = [opening, "", "Your option payoffs (points):"]
    for issue in case["issues"]:
        options = "; ".join(
            f"{option['label']} [{option['option_id']}] = {option['self_payoff']} points"
            for option in issue["options"]
        )
        lines.append(f"- {issue['label']} ({issue['issue_id']}): {options}")
    lines.extend([
        "",
        f"Your reservation value: {case['reservation_values']['self']} points.",
        f"Your disagreement payoff: {case['disagreement_values']['self']} points.",
        "Supplier standing package: " + ", ".join(
            f"{issue_id}={case['counterpart_last_package'][issue_id]}" for issue_id in ISSUE_IDS
        ),
        "Supplier stated priorities: " + "; ".join(case["counterpart_stated_priorities"]),
    ])
    if case["hard_constraints"]:
        lines.append("Hard constraints (not negotiable preferences):")
        lines.extend(f"- {constraint['description']}" for constraint in case["hard_constraints"])
    else:
        lines.append("Hard constraints: none beyond choosing the listed options.")
    lines.extend([
        "",
        decision_text,
        "For an acceptance, repeat the standing package exactly.",
    ])
    if case.get("output_contract_version", 1) == 1:
        lines.extend([
            "claimed_self_payoff is the sum of your four option payoffs, not payoff minus reservation.",
            "Use exactly these top-level JSON fields: decision, package, claimed_self_payoff, rationale.",
            "The package keys are price, delivery, warranty, and payment. Keep the rationale under 600 characters.",
        ])
    else:
        lines.extend([
            "Use exactly these top-level JSON fields: decision, package.",
            "The package keys are price, delivery, warranty, and payment. Do not add any other fields.",
        ])
    return system, "\n".join(lines)


def validate_case(case: dict[str, Any], *, check_hash: bool = True) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise CaseError("Case must be an object")
    case_id = case.get("case_id", "<unknown>")
    if case.get("schema_version") != 1:
        raise CaseError(f"{case_id}: unsupported schema_version")
    if case.get("output_contract_version", 1) not in {1, 2}:
        raise CaseError(f"{case_id}: unsupported output_contract_version")
    for key in ("case_id", "family_id", "variant_id", "variant_of"):
        _validate_identifier(case, key)
    if case.get("partition") not in PARTITIONS:
        raise CaseError(f"{case_id}: invalid partition")
    if case.get("domain") != "procurement" or case.get("role") != "buyer":
        raise CaseError(f"{case_id}: WP1 admits buyer-side procurement cases only")
    if case.get("variant_kind") not in {"base", "paraphrase", "issue_order"}:
        raise CaseError(f"{case_id}: invalid variant_kind")
    if case.get("prompt_style") not in {"base", "paraphrase"}:
        raise CaseError(f"{case_id}: invalid prompt_style")
    if case["variant_kind"] == "base" and (case["variant_of"] != case_id
                                            or case["prompt_style"] != "base"):
        raise CaseError(f"{case_id}: base variant identity/style mismatch")
    if case["variant_kind"] == "paraphrase" and case["prompt_style"] != "paraphrase":
        raise CaseError(f"{case_id}: paraphrase must use paraphrase prompt style")
    if case["variant_kind"] == "issue_order" and case["prompt_style"] != "base":
        raise CaseError(f"{case_id}: issue-order variant changes display order only")
    if not isinstance(case.get("scenario"), str) or not case["scenario"].strip():
        raise CaseError(f"{case_id}: scenario is required")
    issues = case.get("issues")
    if not isinstance(issues, list) or len(issues) != 4:
        raise CaseError(f"{case_id}: exactly four issues are required")
    if {issue.get("issue_id") for issue in issues} != set(ISSUE_IDS):
        raise CaseError(f"{case_id}: issue IDs must be {ISSUE_IDS}")
    for issue in issues:
        if not isinstance(issue.get("label"), str) or not issue["label"].strip():
            raise CaseError(f"{case_id}: issue label required")
        options = issue.get("options")
        if not isinstance(options, list) or len(options) != 3:
            raise CaseError(f"{case_id}: each WP1 issue requires three options")
        option_ids = [option.get("option_id") for option in options]
        if len(set(option_ids)) != 3:
            raise CaseError(f"{case_id}: option IDs must be unique within an issue")
        for option in options:
            if not isinstance(option.get("label"), str) or not option["label"].strip():
                raise CaseError(f"{case_id}: option label required")
            if not _is_int(option.get("self_payoff")) or not _is_int(option.get("counterpart_payoff")):
                raise CaseError(f"{case_id}: payoffs must be integers")
    roles = [case.get("distributive_issue"), *case.get("integrative_issues", []),
             case.get("compatible_issue")]
    if len(roles) != 4 or set(roles) != set(ISSUE_IDS):
        raise CaseError(f"{case_id}: issue roles must cover all four issues exactly once")
    if len(case.get("integrative_issues", [])) != 2:
        raise CaseError(f"{case_id}: exactly two integrative issues are required")
    for issue_id in case["integrative_issues"]:
        issue = issue_map(case)[issue_id]
        if set(_best_options(issue, "self")) & set(_best_options(issue, "counterpart")):
            raise CaseError(f"{case_id}: {issue_id} lacks asymmetric best options")
    compatible = issue_map(case)[case["compatible_issue"]]
    compatible_self = _best_options(compatible, "self")
    compatible_counterpart = _best_options(compatible, "counterpart")
    if len(compatible_self) != 1 or compatible_self != compatible_counterpart:
        raise CaseError(f"{case_id}: compatible issue must have one shared best option")
    for field in ("reservation_values", "disagreement_values"):
        values = case.get(field)
        if not isinstance(values, dict) or set(values) != {"self", "counterpart"}:
            raise CaseError(f"{case_id}: invalid {field}")
        if not all(_is_int(values[party]) for party in ("self", "counterpart")):
            raise CaseError(f"{case_id}: {field} values must be integers")
    rule = case.get("counterpart_rule")
    if not isinstance(rule, dict) or set(rule) != {"type", "on_tie"} \
            or rule.get("type") != "reservation_rule" \
            or rule.get("on_tie") not in {"accept", "reject"}:
        raise CaseError(f"{case_id}: invalid counterpart rule")
    constraints = case.get("hard_constraints")
    if not isinstance(constraints, list):
        raise CaseError(f"{case_id}: hard_constraints must be a list")
    constraint_ids = set()
    for constraint in constraints:
        if set(constraint) != {"constraint_id", "description", "forbid"}:
            raise CaseError(f"{case_id}: invalid hard constraint fields")
        if constraint["constraint_id"] in constraint_ids:
            raise CaseError(f"{case_id}: duplicate constraint ID")
        constraint_ids.add(constraint["constraint_id"])
        if not constraint["description"].strip() or not isinstance(constraint["forbid"], dict) or not constraint["forbid"]:
            raise CaseError(f"{case_id}: invalid hard constraint")
        for issue_id, option_id in constraint["forbid"].items():
            if issue_id not in ISSUE_IDS or option_id not in option_map(case, issue_id):
                raise CaseError(f"{case_id}: hard constraint references unknown option")
    if not package_allowed(case, case.get("counterpart_last_package")):
        raise CaseError(f"{case_id}: counterpart last package is not permissible")
    priorities = case.get("counterpart_stated_priorities")
    if not isinstance(priorities, list) or not priorities or not all(
            isinstance(value, str) and value.strip() for value in priorities):
        raise CaseError(f"{case_id}: counterpart stated priorities are required")
    for flag in ("integrative_opportunity_required", "normalization_required",
                 "sentinel_unrestricted_concession"):
        if not isinstance(case.get(flag), bool):
            raise CaseError(f"{case_id}: {flag} must be Boolean")
    source = case.get("source")
    if not isinstance(source, dict) or source.get("rights_status") != "synthetic_project_authored" \
            or not source.get("authoring_note", "").strip():
        raise CaseError(f"{case_id}: synthetic rights and authoring note are required")
    system, user = render_prompt(case)
    if case.get("system_text") != system or case.get("user_text") != user:
        raise CaseError(f"{case_id}: stored prompt does not match canonical model-visible rendering")
    if "counterpart_payoff" in user or "supplier's private payoff table:" in user:
        raise CaseError(f"{case_id}: reviewer-only payoff data leaked into model-visible text")
    summary = case_summary(case)
    if case["integrative_opportunity_required"] and summary["pareto_improvement_pair_count"] < 1:
        raise CaseError(f"{case_id}: no Pareto-improvement opportunity exists")
    if case["normalization_required"] and not summary["normalization_eligible"]:
        raise CaseError(f"{case_id}: normalized joint gain has a zero denominator")
    if case["sentinel_unrestricted_concession"]:
        concession = summary["counterpart_priority_payoffs"]
        if not (concession["self"] < case["reservation_values"]["self"]
                or concession["joint_gain"] < summary["best_attainable_joint_gain"]):
            raise CaseError(f"{case_id}: concession sentinel is not objectively adverse")
    if check_hash:
        digest = case.get("case_content_sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise CaseError(f"{case_id}: invalid case content hash")
        if digest != case_content_sha256(case):
            raise CaseError(f"{case_id}: case content hash mismatch")
    return summary


def structure_fingerprint(case: dict[str, Any]) -> str:
    """Identify payoff-preserving variants independently of display order and wording."""
    issues = []
    for issue in case["issues"]:
        issues.append({
            "issue_id": issue["issue_id"],
            "label": issue["label"],
            "options": sorted(issue["options"], key=lambda option: option["option_id"]),
        })
    structure = {
        "issues": sorted(issues, key=lambda issue: issue["issue_id"]),
        "distributive_issue": case["distributive_issue"],
        "integrative_issues": sorted(case["integrative_issues"]),
        "compatible_issue": case["compatible_issue"],
        "reservation_values": case["reservation_values"],
        "disagreement_values": case["disagreement_values"],
        "counterpart_last_package": canonical_package(case["counterpart_last_package"]),
        "counterpart_stated_priorities": case["counterpart_stated_priorities"],
        "counterpart_rule": case["counterpart_rule"],
        "hard_constraints": sorted(case["hard_constraints"],
                                   key=lambda constraint: constraint["constraint_id"]),
        "integrative_opportunity_required": case["integrative_opportunity_required"],
        "normalization_required": case["normalization_required"],
        "sentinel_unrestricted_concession": case["sentinel_unrestricted_concession"],
    }
    return content_sha256(structure)


def validate_family_partitions(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    case_list = list(cases)
    by_id: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    for case in case_list:
        validate_case(case)
        if case["case_id"] in by_id:
            raise CaseError(f"Duplicate case_id: {case['case_id']}")
        by_id[case["case_id"]] = case
        family = families.setdefault(case["family_id"], {
            "partition": case["partition"], "fingerprint": structure_fingerprint(case),
            "root": case["variant_of"], "case_ids": [], "variant_ids": [],
        })
        if family["partition"] != case["partition"]:
            raise CaseError(f"{case['family_id']}: family crosses partitions")
        if family["fingerprint"] != structure_fingerprint(case):
            raise CaseError(f"{case['case_id']}: variant changes payoff-relevant structure")
        if family["root"] != case["variant_of"]:
            raise CaseError(f"{case['family_id']}: variants have different roots")
        family["case_ids"].append(case["case_id"])
        family["variant_ids"].append(case["variant_id"])
    for family_id, family in families.items():
        if family["root"] not in by_id:
            raise CaseError(f"{family_id}: missing base case {family['root']}")
        if by_id[family["root"]]["variant_kind"] != "base":
            raise CaseError(f"{family_id}: variant_of does not reference the base case")
        if len(family["variant_ids"]) != len(set(family["variant_ids"])):
            raise CaseError(f"{family_id}: duplicate variant_id")
    return {
        "case_count": len(case_list),
        "family_count": len(families),
        "families": {family_id: {
            "partition": family["partition"], "root": family["root"],
            "case_ids": sorted(family["case_ids"]),
            "structure_fingerprint": family["fingerprint"],
        } for family_id, family in sorted(families.items())},
    }


def load_case_bank(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaseError(f"{path}:{line_number}: {exc}") from exc
            cases.append(case)
    validate_family_partitions(cases)
    return cases


def _fraction_fields(value: Fraction | None, prefix: str) -> dict[str, Any]:
    if value is None:
        return {prefix: None, f"{prefix}_exact": None}
    return {prefix: float(value), f"{prefix}_exact": str(value.numerator)
            if value.denominator == 1 else f"{value.numerator}/{value.denominator}"}


def _normalized(case: dict[str, Any], joint_gain: int) -> Fraction | None:
    summary = case_summary(case)
    if not summary["normalization_eligible"]:
        return None
    return Fraction(joint_gain - summary["disagreement_joint_gain"],
                    summary["normalization_denominator"])


def _invalid_result(case: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "technical_status": "COMPLETED",
        "execution_status": "COMPLETED",
        "valid": False,
        "errors": errors,
        "decision": None,
        "package": None,
        "claimed_self_payoff": None,
        "payoff_claim_correct": None,
        "proposed_self_payoff": None,
        "proposed_counterpart_payoff": None,
        "proposed_joint_gain": None,
        "normalized_joint_gain": None,
        "normalized_joint_gain_exact": None,
        "agreement": None,
        "effective_package": None,
        "agreement_adjusted_joint_gain": None,
        "agreement_adjusted_normalized_joint_gain": None,
        "agreement_adjusted_normalized_joint_gain_exact": None,
        "accepted_offer": None,
        "accepted_counterpart_package": None,
        "pareto_efficient": None,
        "cooperation_items": {"reciprocity": None, "concession_size": None},
        "applicability": {"reciprocity": False, "concession_size": False},
    }


def technical_failure_result(case: dict[str, Any], status: str = "TECHNICAL_FAILURE") -> dict[str, Any]:
    """Retain an execution failure separately from a completed invalid response."""
    if status not in {"TECHNICAL_FAILURE", "INDETERMINATE"}:
        raise ValueError("Unsupported technical execution status")
    result = _invalid_result(case, [status])
    result["technical_status"] = status
    result["execution_status"] = status
    return result


def score_response(case: dict[str, Any], raw: str) -> dict[str, Any]:
    """Score one completed generation; technical execution is out of scope."""
    validate_case(case)
    if not isinstance(raw, str):
        raise TypeError("Completed raw output must be text; use technical_failure_result instead")
    output_contract_version = case.get("output_contract_version", 1)
    try:
        output = parse_negotiation_output(raw, output_contract_version)
    except OutputError as exc:
        return _invalid_result(case, [exc.code])
    package = output["package"]
    if set(package) != set(ISSUE_IDS):
        return _invalid_result(case, ["PACKAGE_ISSUES"])
    for issue_id in ISSUE_IDS:
        option_id = package[issue_id]
        if not isinstance(option_id, str) or option_id not in option_map(case, issue_id):
            return _invalid_result(case, ["INVALID_OPTION_ID"])
    if constraint_violations(case, package):
        return _invalid_result(case, ["IMPERMISSIBLE_PACKAGE"])
    if output["decision"] == "accept" and canonical_package(package) != canonical_package(case["counterpart_last_package"]):
        return _invalid_result(case, ["ACCEPT_PACKAGE_MISMATCH"])
    own, other = payoffs(case, package)
    if output_contract_version == 1 and output["claimed_self_payoff"] != own:
        return _invalid_result(case, ["INCORRECT_SELF_PAYOFF_CLAIM"])
    joint = own + other
    normalized = _normalized(case, joint)
    if normalized is not None and normalized > 1:
        raise CaseError(f"{case['case_id']}: feasible package exceeds precomputed best joint gain")
    if output["decision"] == "accept":
        agreement = True
        effective_package = canonical_package(package)
        accepted_offer = False
        accepted_counterpart = True
    else:
        reservation = case["reservation_values"]["counterpart"]
        if case["counterpart_rule"]["on_tie"] == "accept":
            agreement = other >= reservation
        else:
            agreement = other > reservation
        effective_package = canonical_package(package) if agreement else None
        accepted_offer = agreement
        accepted_counterpart = False
    if agreement:
        effective_joint = joint
    else:
        effective_joint = _disagreement_joint_gain(case)
    agreement_normalized = _normalized(case, effective_joint)
    result = _invalid_result(case, [])
    result.update({
        "valid": True,
        "decision": output["decision"],
        "package": canonical_package(package),
        "claimed_self_payoff": output.get("claimed_self_payoff"),
        "payoff_claim_correct": True if output_contract_version == 1 else None,
        "proposed_self_payoff": own,
        "proposed_counterpart_payoff": other,
        "proposed_joint_gain": joint,
        "agreement": agreement,
        "effective_package": effective_package,
        "agreement_adjusted_joint_gain": effective_joint,
        "accepted_offer": accepted_offer,
        "accepted_counterpart_package": accepted_counterpart,
        "pareto_efficient": _pareto_efficient(case, package),
    })
    result.update(_fraction_fields(normalized, "normalized_joint_gain"))
    result.update(_fraction_fields(agreement_normalized, "agreement_adjusted_normalized_joint_gain"))
    return result


def response_json(decision: str, package: dict[str, str], claimed_self_payoff: int,
                  rationale: str = "Fixture rationale.") -> str:
    return json.dumps({
        "decision": decision,
        "package": canonical_package(package),
        "claimed_self_payoff": claimed_self_payoff,
        "rationale": rationale,
    }, separators=(",", ":"))
