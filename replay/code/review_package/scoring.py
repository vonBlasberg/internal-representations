"""Compatibility guard around the preserved negotiation scorer. No response repair."""
import json
from .reuse import legacy


def score_completed(scoring, case, raw):
    try:
        return scoring.score_response(case, raw)
    except TypeError:
        value = json.loads(raw, object_pairs_hook=legacy._unique_object)
        if (isinstance(value, dict) and set(value) == {"decision", "package"}
                and isinstance(value["decision"], (list, dict))):
            return scoring._invalid_result(case, ["DECISION"])
        raise
