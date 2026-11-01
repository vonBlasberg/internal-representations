"""Notebook presentation only. Scientific analysis is the unmodified copied code."""
if not __debug__:
    raise RuntimeError("Run verification without Python -O or PYTHONOPTIMIZE; required checks must stay enabled.")

from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import shutil
from statistics import mean
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "replay"
LABELS = {"unchanged": "Baseline", "internal": "Internal edit", "prompt": "Prompting",
          "sham": "Zero edit", "random_1": "Random 1", "random_2": "Random 2",
          "random_3": "Random 3", "reverse": "Reverse", "permuted": "Permuted"}
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "normal", "axes.titlesize": 10,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources():
    manifest = read(ROOT / "source_manifest.json")
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundled source changed: {name}")
    return {"files_checked": len(manifest["sha256"]), "status": "All bundled inputs match", "model_runs": 0}


def recalculate():
    """Retain each replay in a new directory; refresh derived reader exports."""
    verify_sources()
    (ROOT / "outputs").mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="notebook-", dir=ROOT / "outputs"))
    env = {**os.environ, "PYTHONPATH": str(REPLAY / "code"), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1"}
    for attempt in ("a05", "a06", "a09", "a10"):
        subprocess.run([sys.executable, "-B", "-m", "review_package.cli", "history", "--attempt", attempt,
                        "--output", str(destination / attempt)], env=env, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([sys.executable, "-B", "-m", "review_package.cli", "compare-saved", "--manifest",
                    str(REPLAY / "comparison/delivery/manifest.json"), "--allow-real-saved-records",
                    "--output", str(destination / "comparison")], env=env, check=True, capture_output=True, text=True, timeout=120)
    comparison = read(destination / "comparison/review.json")
    expected = read(REPLAY / "comparison/report/review.json")
    for key in expected:
        if key != "analysis_provenance" and comparison[key] != expected[key]:
            raise AssertionError(f"Recalculation disagrees with saved result: {key}")
    spec = importlib.util.spec_from_file_location("independent_payoff_check", REPLAY / "verify_findings.py")
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    audit = independent.check(destination / "comparison/review.json", REPLAY / "generation_inputs/request.json")
    (destination / "independent_check.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Stable download links for readers of the saved notebook. Fresh run folders
    # above retain the complete recalculation; these are derived convenience copies.
    exports = ROOT / "exports"
    exports.mkdir(exist_ok=True)
    for source, target in ((destination / "comparison/all_attempts.csv", "evaluation_attempts.csv"),
                           (destination / "a10/all_attempts.csv", "calibration_attempts.csv"),
                           (destination / "independent_check.json", "independent_check.json")):
        shutil.copyfile(source, exports / target)
    admission = read(REPLAY / "comparison/audit.json")
    request = read(REPLAY / "generation_inputs/request.json")
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert admission["status"] == "FIXED_COMPARISON_INDEPENDENTLY_VERIFIED"
    assert admission["request_sha256"] == digest and admission["attempts"] == len(comparison["all_trials"]) == 324
    published = {**comparison, "historical_evidence_label": comparison["evidence_label"],
                 "evidence_label": admission["status"],
                 "admission_record": "replay/comparison/audit.json",
                 "admission_record_sha256": hashlib.sha256((REPLAY / "comparison/audit.json").read_bytes()).hexdigest()}
    (exports / "findings.json").write_text(json.dumps(published, indent=2) + "\n")
    descriptive = {"scope": "Descriptive comparison; frozen primary/secondary contrasts unchanged",
                   "outcome": "agreement_adjusted_joint_gain",
                   "weighting": "Equal families; equal variants within family",
                   "missing_outcomes": "Feasible accepted-package and disagreement-payoff bounds",
                   "source_request_sha256": hashlib.sha256((REPLAY / "generation_inputs/request.json").read_bytes()).hexdigest(),
                   "source_report_sha256": audit["source_report_sha256"],
                   "contrasts": audit["descriptive_contrasts"]}
    (exports / "descriptive_comparisons.json").write_text(json.dumps(descriptive, indent=2) + "\n", encoding="utf-8")
    party_payoffs(comparison).to_csv(exports / "party_payoff_differences.csv", index=False)
    return {"output": destination, "comparison": comparison,
            "calibration": read(destination / "a10/review.json"),
            "history": [read(destination / n / "review.json") for n in ("a05", "a06", "a09", "a10")],
            "independent_check": audit}


def table(rows):
    return pd.DataFrame(rows).fillna("Not defined")


def repeatability(output_dir=None):
    """Recount audited repeats from response records; do not rerun GPU checks."""
    root = REPLAY / "repeatability"
    references = {r["original_trial_id"]: r for r in read(root / "reference.json")["rows"]}
    plan = read(root / "plan.json")
    fields = plan["analysis"]["behavior_fields"]
    summary = read(root / "combined/summary.json")
    comparison = read(root / "combined/comparison.json")
    rows = read(root / "combined/outcomes.json")
    assert len(rows) == len({r["trial_id"] for r in rows}) == 72
    assert len(references) == 36
    details = {r["trial_id"]: r for r in comparison["outputs"]}
    assert set(details) == {r["trial_id"] for r in rows}
    sessions = summary["sessions"]
    assert len(sessions) == len({s["audited_call_ids"][0] for s in sessions}) == 2
    exports, display_rows = [], []
    for session in sessions:
        sid = session["audited_session_ids"][0]
        current = [r for r in rows if r["session_id"] == sid]
        assert len(current) == 36 and {r["original_trial_id"] for r in current} == set(references)
        counts = {"behavior_matches": 0, "text_matches": 0, "tokens_match": 0, "processed_logits_match": 0}
        for row in current:
            old, checked = references[row["original_trial_id"]], details[row["trial_id"]]
            matches = {"behavior_matches": all(row["score"][key] == old["score"][key] for key in fields),
                       "text_matches": row["raw_response"] == old["raw_response"],
                       "tokens_match": row["generated_token_ids"] == old["generated_token_ids"],
                       "processed_logits_match": row["processed_logits_match"]}
            assert all(matches[key] == checked[key] for key in matches)
            for key, value in matches.items():
                counts[key] += value
            exports.append({"trial_id": row["trial_id"], "session_id": sid,
                            "original_trial_id": row["original_trial_id"], "condition": old["condition"],
                            **matches, "raw_response": row["raw_response"],
                            "score": json.dumps(row["score"], sort_keys=True),
                            "generated_token_ids": json.dumps(row["generated_token_ids"])})
        contrasts = [c for c in comparison["contrasts"] if c["session_id"] == sid]
        assert len(contrasts) == len({c["case_id"] for c in contrasts}) == 12
        matching = sum(c["contrast"] == c["reference_contrast"] for c in contrasts)
        assert matching == session["matching_contrasts"]
        for field, reported in (("behavior_matches", "behavior_matching_outputs"), ("text_matches", "text_matching_outputs"),
                                ("tokens_match", "token_matching_outputs"), ("processed_logits_match", "logit_matching_outputs")):
            assert counts[field] == session[reported]
        display_rows.append({"Session": "Fresh session 1" if "R01" in sid else "Fresh session 2",
                             "Behavior": f"{counts['behavior_matches']}/36", "Text": f"{counts['text_matches']}/36",
                             "Tokens": f"{counts['tokens_match']}/36", "Processed logits": f"{counts['processed_logits_match']}/36",
                             "Edit contrasts": f"{matching}/12"})
    assert sum(r["behavior_matches"] for r in exports) == summary["behavior_matching_outputs"]
    assert sum(s["matching_contrasts"] for s in sessions) == summary["matching_contrasts"]
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "repeatability_outputs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exports[0]))
            writer.writeheader()
            writer.writerows(exports)
        shutil.copy2(root / "combined/summary.json", destination / "repeatability_summary.json")
    return table(display_rows)


def party_payoffs(comparison):
    """Descriptive equal-family means on valid pairs, not missing-value imputation."""
    baseline = {r["case_id"]: r for r in comparison["all_trials"] if r["condition"] == "unchanged"}
    fields = {"Proposed buyer": "proposed_self_payoff", "Proposed supplier": "proposed_counterpart_payoff",
              "Buyer after rule": "agreement_adjusted_self_payoff",
              "Supplier after rule": "agreement_adjusted_counterpart_payoff"}
    results = []
    for condition in LABELS:
        if condition == "unchanged":
            continue
        rows = [r for r in comparison["all_trials"] if r["condition"] == condition
                and r["score"]["valid"] and baseline[r["case_id"]]["score"]["valid"]]
        values = {}
        for label, field in fields.items():
            groups = {}
            for row in rows:
                groups.setdefault(row["family_id"], []).append(
                    row["score"][field] - baseline[row["case_id"]]["score"][field])
            values[label] = mean(mean(v) for v in groups.values()) if groups else None
        results.append({"Condition": LABELS[condition], "Valid pairs": len(rows), **values})
    return table(results)


def construction_example():
    """Show the first saved pair, checking its payoffs directly against utility tables."""
    request = read(REPLAY / "evidence/a10-request.json")
    pair = request["construction_source"]["references"]["pairs"][0]
    case = next(c for c in request["construction_request"]["program"]["cases"]["construction_fit"]
                if c["case_id"] == pair["case_id"])
    rows = []
    for kind, label in (("negative", "Accept standing offer"), ("positive", "Mutually improving counteroffer")):
        raw = pair[kind + "_text"]
        value = json.loads(raw.replace("\\n", "\n"))
        payoffs = {}
        for party, key in (("Buyer", "self"), ("Supplier", "counterpart")):
            payoffs[party] = sum(next(o[key + "_payoff"] for o in issue["options"]
                                     if o["option_id"] == value["package"][issue["issue_id"]])
                                 for issue in case["issues"])
            assert payoffs[party] == pair["author_only_scores"][kind]["proposed_" + key + "_payoff"]
        rows.append({"Reference": label, "Decision": value["decision"],
                     "Package": json.dumps(value["package"], sort_keys=True), **payoffs,
                     "Combined": payoffs["Buyer"] + payoffs["Supplier"]})
    return case["scenario"], table(rows)


def input_prompts(case_id):
    """Return the exact archived starting messages, not a reconstructed prompt."""
    plan = read(REPLAY / "comparison/delivery/original-plan.json")
    return {condition: next(t["execution_input"]["messages"] for t in plan["trials"]
                            if t["case_id"] == case_id and t["condition"] == condition)
            for condition in ("unchanged", "internal", "prompt")}


def candidates(calibration):
    return table([{"Layer (zero-based)": r["edit"]["layer"], "Strength": r["edit"]["coefficient"],
                   "Valid": r["score"]["valid"], "Buyer change": r["decision_and_payoffs"]["buyer_delta"],
                   "Supplier change": r["decision_and_payoffs"]["supplier_delta"],
                   "Joint change": r["decision_and_payoffs"]["joint_delta"],
                   "Qualifies": r["historical_rule_qualified"], "Selected": r["historically_selected"],
                   "Why": " ".join(r["selection_reasons"])}
                  for r in calibration["rows"] if r["kind"] == "internal"])


def save(fig, name):
    destination = ROOT / "outputs/figures"
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    return fig


def condition_table(comparison):
    return table([{"Condition": LABELS.get(r["condition"], r["condition"]), "Attempts": r["planned"],
                   "Valid": r["valid"], "Invalid": r["invalid"], "Counteroffers": r["offers"],
                   "Agreements": r["simulated_agreements"], "Proposed buyer payoff below minimum": r["buyer_below_reservation"]}
                  for r in comparison["accounting"]])


def contrast_table(comparison):
    def interval(value):
        return "Not defined" if value is None else f"[{value[0]:.2f}, {value[1]:.2f}]"
    return table([{"Comparison": LABELS[c["left"]] + " minus " + LABELS[c["right"]],
                   "Valid pairs / 36": c["valid_pairs"],
                   "Mean difference": round(c["point_estimate"], 2) if c["point_estimate"] is not None else "Not identified",
                   "Missing-outcome bounds": interval(c["full_case_bank_invalid_response_bounds"]),
                   "Family-bootstrap percentile interval": interval(c["case_bank_sensitivity_interval"]),
                   "Variation note": "No variation across families in this case bank" if c["no_observed_family_variation"] else ""}
                  for c in comparison["contrasts"]])


def outcome_figure(comparison):
    order = ("unchanged", "internal", "prompt", "sham", "random_1", "random_2", "random_3", "reverse", "permuted")
    rows = [next(r for r in comparison["accounting"] if r["condition"] == c) for c in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.7), sharey=True, layout="constrained")
    for ax, field, title in zip(axes, ("offers", "simulated_agreements", "invalid"),
                               ("Counteroffers", "Agreements", "Invalid answers")):
        for i, row in enumerate(rows):
            value = row[field]
            ax.plot([0, value], [i, i], color="black", lw=.7)
            ax.scatter(value, i, s=28, facecolors="black" if i < 3 else "white", edgecolors="black", linewidths=.8, zorder=3)
            ax.annotate(str(value), (value, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set(xlim=(-1, 41), xticks=[0, 12, 24, 36], title=title)
        ax.axhline(2.5, color=".75", lw=.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), [LABELS[r["condition"]] for r in rows])
    axes[0].set_ylim(len(rows)-.5, -.6)
    fig.supxlabel("Number of cases, out of 36 attempts per condition", fontsize=9)
    return save(fig, "02_decisions_and_agreements")


def bounds_figure(comparison):
    rows = [c for c in comparison["contrasts"] if c["left"] == "internal" and c["right"] in {"unchanged", "prompt"}]
    fig, ax = plt.subplots(figsize=(7.7, 2.0), layout="constrained")
    for i, row in enumerate(rows):
        lower, upper = row["full_case_bank_invalid_response_bounds"]
        ax.plot([lower, upper], [i, i], color="black", lw=1.4)
        ax.scatter([lower, upper], [i, i], s=22, facecolors="white", edgecolors="black", linewidths=.7, zorder=3)
        ax.text((lower+upper)/2, i-.18, f"{lower:.2f} to {upper:.2f}", ha="center", fontsize=9)
    ax.axvline(0, color=".5", lw=.7, linestyle=":")
    ax.set(xlim=(-52, 5), ylim=(1.5, -.55), xticks=[-50, -40, -30, -20, -10, 0])
    ax.set_yticks(range(len(rows)), [f"Edit minus {LABELS[r['right']].lower()}" for r in rows])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Combined payoff difference after simulated acceptance (points)",
           title="Missing-outcome bounds, not confidence intervals",)
    return save(fig, "03_payoff_bounds")


def wording_figure(comparison):
    rows = sorted([r for r in comparison["variant_sensitivity"] if r["left"] == "internal" and r["right"] == "unchanged"
                   and r["outcome"] == "agreement_adjusted_joint_gain"], key=lambda r: r["family_id"])
    variants = ("base", "paraphrase", "issue_order")
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 4.1), sharey=True, layout="constrained")
    for ax, variant, title in zip(axes, variants, ("Original", "Paraphrased", "Issues reordered")):
        for i, row in enumerate(rows):
            value = next(v["joint_delta"] for v in row["variants"] if v["variant"] == variant)
            if value is None:
                ax.text(-62, i, "invalid", ha="center", va="center", fontsize=8)
            else:
                ax.plot([value, 0], [i, i], color="black", lw=.7)
                ax.scatter(value, i, s=25, facecolors="white" if value == 0 else "black", edgecolors="black", linewidths=.7, zorder=3)
                ax.annotate(f"{value:g}", (value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": .4})
        ax.axvline(0, color=".7", lw=.5, zorder=0)
        ax.set(xlim=(-145, 23), xticks=[-120, -60, 0], title=title)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), [r["family_id"].removeprefix("prospective-20261031-").capitalize() for r in rows])
    axes[0].set_ylim(len(rows)-.5, -.6)
    fig.supxlabel("Edit minus baseline: combined payoff after simulated acceptance (points)", fontsize=9)
    return save(fig, "04_wording_sensitivity")


def inspect_case(comparison, case_id):
    cases = {c["case_id"]: c for c in read(REPLAY / "comparison/delivery/cases.json")}
    case = cases[case_id]; output=[]
    labels = {(issue["issue_id"], opt["option_id"]): opt["label"] for issue in case["issues"] for opt in issue["options"]}
    for condition in ("unchanged", "internal", "prompt"):
        row=next(r for r in comparison["all_trials"] if r["case_id"] == case_id and r["condition"] == condition)
        score=row["score"]
        output.append({"Condition": LABELS[condition], "Valid": score["valid"], "Decision": score["decision"],
                       "Terms": "; ".join(labels[(k,v)] for k,v in (score["package"] or {}).items()),
                       "Buyer if agreed": score["proposed_self_payoff"], "Supplier if agreed": score["proposed_counterpart_payoff"],
                       "Agreement": score["agreement"], "Buyer after rule": score["agreement_adjusted_self_payoff"],
                       "Supplier after rule": score["agreement_adjusted_counterpart_payoff"]})
    return case["scenario"], table(output)


def case_figure(comparison, case_id):
    _, frame = inspect_case(comparison, case_id)
    case = next(c for c in read(REPLAY / "comparison/delivery/cases.json") if c["case_id"] == case_id)
    fig, axes = plt.subplots(1,2, figsize=(7.7, 3.2), layout="constrained", sharey=True)
    values = pd.to_numeric(frame[["Buyer if agreed", "Supplier if agreed", "Buyer after rule", "Supplier after rule"]].stack(), errors="coerce")
    upper = max(100, float(values.max()) + 15) if values.notna().any() else 100
    lower = min(0, float(values.min()) - 10) if values.notna().any() else 0
    for ax, party in zip(axes, ("Buyer", "Supplier")):
        for i, row in frame.iterrows():
            proposed, realized = pd.to_numeric(pd.Series([row[f"{party} if agreed"], row[f"{party} after rule"]]), errors="coerce")
            if not np.isfinite(proposed) or not np.isfinite(realized):
                ax.text(upper/2, i, "invalid", ha="center", va="center")
                continue
            ax.plot([proposed, realized], [i, i], color="black", lw=.7)
            ax.scatter(proposed, i, s=52, facecolors="white", edgecolors="black", linewidths=.8, zorder=3)
            ax.scatter(realized, i, s=16, color="black", zorder=4)
            for value in set((proposed, realized)):
                ax.annotate(f"{value:g}", (value, i), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8)
        ax.set(xlim=(lower-4, upper), title=party, xlabel="Payoff points")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(frame)), frame["Condition"])
    axes[0].set_ylim(len(frame)-.25, -.6)
    family = case["family_id"].removeprefix("prospective-20261031-").capitalize()
    wording = {"base": "original wording", "paraphrase": "paraphrased", "issue_order": "issues reordered"}[case["variant_kind"]]
    fig.suptitle(f"{family}: {wording}\nOpen circle: payoff if agreed    Solid dot: after supplier rule", fontsize=9)
    threshold = case["reservation_values"]["counterpart"]
    relation = "at least" if case["counterpart_rule"]["on_tie"] == "accept" else "more than"
    disagreement = case["disagreement_values"]
    fig.supxlabel(f"Counteroffers require {relation} {threshold:g} supplier points.\n"
                  f"Rejection payoffs: buyer {disagreement['self']:g}, supplier {disagreement['counterpart']:g}.  Case: {case_id}", fontsize=8)
    return save(fig, "05_example_payoffs")


def case_interpretation(comparison, case_id):
    """Describe the displayed saved case, including invalid and rejected answers."""
    _, frame = inspect_case(comparison, case_id)
    case = next(c for c in read(REPLAY / "comparison/delivery/cases.json") if c["case_id"] == case_id)
    threshold = case["reservation_values"]["counterpart"]
    relation = "at least" if case["counterpart_rule"]["on_tie"] == "accept" else "more than"
    sentences = []
    for _, row in frame.iterrows():
        label = row["Condition"]
        if not row["Valid"]:
            sentences.append(f"{label} returns an invalid answer, so its payoffs are undefined.")
            continue
        buyer, supplier = row["Buyer after rule"], row["Supplier after rule"]
        if row["Decision"] == "accept":
            sentences.append(f"{label} accepts the standing package, yielding {buyer:g} buyer points and {supplier:g} supplier points.")
        elif row["Agreement"]:
            sentences.append(f"{label} counteroffers and meets the supplier rule, yielding {buyer:g} buyer points and {supplier:g} supplier points.")
        else:
            sentences.append(f"{label} counteroffers, but its {row['Supplier if agreed']:g} supplier points do not meet the requirement of {relation} {threshold:g}. "
                             f"Rejection leaves the buyer with {buyer:g} and the supplier with {supplier:g} points.")
    return " ".join(sentences)


def synthetic_example(budget=100, responses=None, scorer=None):
    sys.path.insert(0, str(REPLAY / "code"))
    from review_package.workflow import Study, AllocationScorer, evaluate_saved
    if type(budget) is not int or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    if responses is None:
        responses = {"baseline": {"team_a": 70, "team_b": 30},
                     "variant": {"team_a": 50, "team_b": 50},
                     "invalid-example": {"team_a": 80, "team_b": 50}}
    study=Study("Synthetic allocation: interface test", ({"case_id":"allocation-demo", "budget":budget},),
                tuple(responses), fixture_only=True)
    authored=[{"case_id":"allocation-demo", "condition":condition, "execution_status":"COMPLETED",
               "raw_response": json.dumps(value) if isinstance(value, dict) else value}
              for condition, value in responses.items()]
    return evaluate_saved(study, scorer if scorer is not None else AllocationScorer(), authored)
