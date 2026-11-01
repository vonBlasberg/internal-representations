"""Flat, self-contained figures and CSVs from the computed analysis, not model calls."""
import csv
import hashlib
from html import escape
import json
from pathlib import Path

from .analyze import LABELS


def number(value):
    return "Not available" if value is None else f"{value:,.4g}"


def _table(headers, rows):
    return ("<div class='table-scroll'><table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _svg_frame(title, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title, quote=True)}" viewBox="0 0 920 {height}">'
            f'<title>{escape(title)}</title><rect width="920" height="{height}" fill="white"/>'
            '<g fill="black" stroke="black" font-family="Arial,sans-serif" font-size="14">'
            + body + '</g></svg>')


def _text(x, y, label, anchor="start"):
    return f'<text x="{x}" y="{y}" stroke="none" text-anchor="{anchor}">{escape(str(label))}</text>'


def _scale(values, start=275, end=770):
    values = [float(v) for v in values if v is not None]
    low, high = min([0, *values]), max([0, *values])
    if high == low:
        low, high = low - 1, high + 1
    padding = (high - low) * .08
    low, high = low - padding, high + padding
    return lambda value: start + (value - low) / (high - low) * (end - start), low, high


def family_figure(analysis):
    rows = analysis["contrasts"][0]["families"]
    scale, low, high = _scale([r[key] for r in rows for key in ("left_mean", "right_mean")])
    bottom = 75 + 44 * len(rows)
    body = _text(18, 24, "Proposed joint gain by family") + _text(275, 45, "Open circle: baseline. Filled circle: internal edit.")
    body += f'<line x1="275" y1="{bottom}" x2="770" y2="{bottom}"/>'
    body += _text(275, bottom + 23, number(low)) + _text(770, bottom + 23, number(high), "end")
    for i, row in enumerate(rows):
        y = 80 + 44 * i
        body += _text(18, y + 5, row["family_id"])
        if row["delta"] is None:
            body += _text(275, y + 5, f"Incomplete: {row['valid_pairs']}/{row['planned_pairs']} valid pairs")
            continue
        a, b = scale(row["right_mean"]), scale(row["left_mean"])
        body += f'<line x1="{a}" x2="{b}" y1="{y}" y2="{y}" stroke-width="2"/>'
        body += f'<circle cx="{a}" cy="{y}" r="5" fill="white" stroke-width="2"/><circle cx="{b}" cy="{y}" r="4"/>'
        body += _text(800, y + 5, f"Change {row['delta']:+.4g}")
    return _svg_frame("Paired family joint gain", bottom + 40, body)


def contrast_figure(analysis):
    rows = analysis["contrasts"]
    values = []
    for row in rows:
        values.extend([row["point_estimate"], *(row["case_bank_sensitivity_interval"] or [])])
    scale, low, high = _scale(values)
    bottom = 85 + 45 * len(rows)
    body = _text(18, 24, "Differences in proposed joint gain")
    body += _text(275, 46, "Intervals describe case-bank sensitivity, not population inference.")
    body += f'<line x1="{scale(0)}" x2="{scale(0)}" y1="60" y2="{bottom}" stroke-dasharray="4 4"/>'
    for i, row in enumerate(rows):
        y = 80 + 45 * i
        body += _text(18, y + 5, f"{LABELS[row['left']]} minus {LABELS[row['right']]}")
        point = row["point_estimate"]
        if point is None:
            body += _text(275, y + 5, "Full-bank estimate unavailable")
            continue
        interval = row["case_bank_sensitivity_interval"]
        if interval:
            a, b = map(scale, interval)
            body += f'<line x1="{a}" x2="{b}" y1="{y}" y2="{y}" stroke-width="2"/>'
        body += f'<circle cx="{scale(point)}" cy="{y}" r="4"/>'
        body += _text(800, y + 5, f"{point:+.4g}")
    body += f'<line x1="275" x2="770" y1="{bottom}" y2="{bottom}"/>'
    body += _text(275, bottom + 23, number(low)) + _text(770, bottom + 23, number(high), "end")
    return _svg_frame("Paired condition contrasts", bottom + 40, body)


def payoff_figure(analysis):
    rows = analysis["payoff_decomposition"]
    keys = ("proposed_self_payoff", "proposed_counterpart_payoff")
    scale, low, high = _scale([r[k] for r in rows for k in keys])
    bottom = 95 + len(rows) * 52
    body = _text(18, 24, "Payoffs to each party") + _text(275, 45, "Black: model/buyer. White: counterpart/supplier.")
    for i, row in enumerate(rows):
        y = 75 + i * 52
        body += _text(18, y + 15, LABELS[row["condition"]])
        if not row["full_case_bank"]:
            body += _text(275, y + 15, f"Incomplete: {row['valid_cases']}/{row['planned_cases']} valid cases")
            continue
        for j, key in enumerate(keys):
            left, right = sorted((scale(0), scale(row[key])))
            body += f'<rect x="{left}" y="{y + 19*j}" width="{max(right-left, .8)}" height="13" fill="{"black" if j == 0 else "white"}"/>'
            body += _text(800, y + 12 + 19*j, number(row[key]))
    body += f'<line x1="275" x2="770" y1="{bottom}" y2="{bottom}"/>'
    body += _text(275, bottom + 23, number(low)) + _text(770, bottom + 23, number(high), "end")
    return _svg_frame("Own and counterpart payoff decomposition", bottom + 40, body)


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    value = "'" + value
                safe[key] = value
            writer.writerow(safe)


def write_report(analysis, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    binding = analysis["provenance"].get("source_binding")
    if binding is not None:
        (directory / "source_binding.json").write_text(json.dumps(binding, indent=2, allow_nan=False) + "\n")
    flat = [{**{k: v for k, v in row.items() if k != "score"}, **row["score"]} for row in analysis["scored_trials"]]
    _csv(directory / "scored_trials.csv", flat)
    _csv(directory / "accounting.csv", analysis["accounting"])
    _csv(directory / "family_contrasts.csv", analysis["contrasts"][0]["families"])
    _csv(directory / "condition_contrasts.csv", [{k: v for k, v in row.items() if k != "families"} for row in analysis["contrasts"]])
    _csv(directory / "payoff_decomposition.csv", analysis["payoff_decomposition"])
    _csv(directory / "secondary_outcomes.csv", [{k: v for k, v in row.items() if k != "families"} for row in analysis["secondary_outcomes"]])
    _csv(directory / "case_interpretations.csv", analysis["case_interpretations"])
    for filename, make in (("family_joint_gain.svg", family_figure), ("condition_contrasts.svg", contrast_figure), ("payoffs.svg", payoff_figure)):
        (directory / filename).write_text(make(analysis))
    fixture = analysis["fixture_only"]
    banner = "SOFTWARE TEST. INVENTED ANSWERS. NO RESEARCH RESULTS." if fixture else "SAVED-OUTPUT ANALYSIS. NOT APPROVED FOR RELEASE."
    primary = analysis["contrasts"][0]
    estimate = number(primary["point_estimate"])
    interval = primary["case_bank_sensitivity_interval"]
    lead = (f"The internal-minus-baseline difference is {estimate} payoff points. "
            if primary["point_estimate"] is not None else "A full-case-bank point estimate is not available. ")
    if fixture:
        lead += "These answers were written to test the analysis. The difference is not a model finding."
    else:
        lead += "Interpretation requires the separate model, study, and source-record checks."
    stat_rows = [
        ["Cases / families", f"{analysis['case_count']} / {analysis['family_count']}"],
        ["Completed / planned pairs", f"{primary['completed_pairs']} / {primary['planned_pairs']}"],
        ["Valid / planned pairs", f"{primary['valid_pairs']} / {primary['planned_pairs']}"],
        ["Internal minus baseline", estimate],
        ["Case-bank sensitivity interval", " to ".join(map(number, interval)) if interval else "Not reported"],
        ["Bounds including invalid answers", "No invalid answers in this contrast" if primary['valid_pairs'] == primary['planned_pairs'] else " to ".join(map(number, primary['full_case_bank_invalid_response_bounds'])) if primary['full_case_bank_invalid_response_bounds'] else "Not available for the full bank"],
    ]
    accounting = _table(["Condition", "Planned", "Completed", "Technical failures", "Indeterminate", "Valid", "Invalid", "Offers", "Accepted offers", "Acceptances"],
        [[LABELS[row['condition']], *[row[key] for key in ("planned_attempts", "completed", "technical_failures", "indeterminate", "valid", "invalid", "offers", "accepted_offers", "accepted_counterpart_packages")]] for row in analysis["accounting"]])
    sections = [
        ("Family results", "family_joint_gain.svg", "family_contrasts.csv", "Each family receives equal weight after averaging its declared variants. Missing or invalid pairs remain visible."),
        ("Comparisons", "condition_contrasts.svg", "condition_contrasts.csv", "Internal minus baseline is primary. Internal minus prompting is the key secondary comparison. Other comparisons are exploratory. No p-values or population claims are produced."),
        ("Consequences for both parties", "payoffs.svg", "payoff_decomposition.csv", "These are proposed payoffs. Agreement and agreement-adjusted joint gain are separate outcomes. A higher combined payoff alone does not establish mutual-benefit seeking."),
    ]
    body = f'<header><p>Beyond the Answer</p><h1>Analysis of saved decisions</h1><p class="banner">{banner}</p></header>'
    body += f'<p>{escape(lead)}</p><p>Phase: {escape(analysis["phase"])}. Model: {escape(analysis["model_id"])}.</p>'
    body += _table(["Primary comparison", "Value"], stat_rows)
    body += '<p>Invalid answers receive finite package-range bounds, not zero payoffs. Technical failures do not count as decisions. Partial valid-pair descriptions are available in the data but never replace the full-bank estimate.</p>'
    for title, image, data, caption in sections:
        body += f'<section><h2>{title}</h2><img src="{image}" alt="{title}. Exact values are in the linked CSV."><p>{caption}</p><a href="{data}">Download figure data</a></section>'
    body += f'<section><h2>Completion and validity</h2>{accounting}<p>Every planned attempt is counted. Validity is distinct from completion and from agreement.</p><a href="accounting.csv">Download counts</a></section>'
    body += '<section><h2>Responses and reproducibility</h2><p>The JSON contains all responses, scores, family calculations, input hashes, and analysis settings. CSVs support independent recalculation.</p><p><a href="analysis.json">Full analysis JSON</a> · <a href="scored_trials.csv">Responses and scores</a> · <a href="secondary_outcomes.csv">Secondary outcomes</a> · <a href="case_interpretations.csv">Case interpretations</a></p>'
    if binding is not None:
        body += '<p>Every analysis row matches its saved source sample, including the exact response, execution status and recorded trial settings. <a href="source_binding.json">View the source checks</a>. Model qualification, evidence review and release approval remain separate.</p>'
    else:
        body += '<p>Checksum verification does not verify that a saved response belongs to its stated source-log sample. That binding, model qualification, and release approval remain separate checks.</p>'
    body += '</section>'
    style = 'body{font:16px/1.55 system-ui,sans-serif;color:#111;background:white;max-width:1040px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{font-size:23px}section{border-top:1px solid #111;margin-top:36px;padding-top:18px}.banner{border:2px solid black;padding:12px;font-weight:700}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #bbb;vertical-align:top}th{border-bottom:2px solid black}.table-scroll{overflow-x:auto}a{color:#111;text-underline-offset:3px}header{border-bottom:2px solid #111;margin-bottom:24px}p{max-width:850px}'
    html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Saved decision analysis</title><style>{style}</style></head><body>{body}</body></html>'
    (directory / "index.html").write_text(html)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
    (directory / "files.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
