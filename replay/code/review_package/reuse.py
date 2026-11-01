"""Load only the copied offline analysis, keeping its original source checks."""
from pathlib import Path
import importlib.metadata
import sys

PACKAGE = Path(__file__).resolve().parents[2]
REUSED = PACKAGE / "code" / "reused"
sys.path.insert(0, str(REUSED))
from submission_analysis import analyze as legacy
from submission_analysis import source_binding as binding

if Path(legacy.__file__).resolve() != REUSED / "submission_analysis" / "analyze.py":
    raise RuntimeError("Unexpected analysis import")
if Path(binding.__file__).resolve() != REUSED / "submission_analysis" / "source_binding.py":
    raise RuntimeError("Unexpected source-binding import")


def protocol(path=None):
    path = Path(path) if path else PACKAGE / "protocol" / "prospective_validation_v2.json"
    value = legacy.load_json(path)
    if value.get("execution_enabled") is not False:
        raise legacy.AnalysisError("This review package must not enable execution")
    if value.get("candidate_reselection_allowed") is not False:
        raise legacy.AnalysisError("Evaluation cannot reselect a candidate")
    if value.get("bootstrap") != {"seed": 20261029, "draws": 10000, "rng": "numpy.PCG64"}:
        raise legacy.AnalysisError("Protocol and retained resampling implementation differ")
    if (value.get("primary_outcome") not in {"proposed_joint_gain", "agreement_adjusted_joint_gain"}
            or value.get("primary_contrast") != ["internal", "unchanged"]
            or value.get("key_secondary_contrast") != ["internal", "prompt"]):
        raise legacy.AnalysisError("Protocol and implemented contrasts differ")
    return value, legacy.sha(path)


def analysis_provenance():
    paths = list((PACKAGE / "code" / "review_package").glob("*.py"))
    paths += list((REUSED / "submission_analysis").glob("*.py"))
    return {"python": sys.version, "numpy": legacy.np.__version__,
            "jsonschema": importlib.metadata.version("jsonschema"),
            "scientific_source_sha256": dict(legacy.PINS),
            "implementation_sha256": {str(path.relative_to(PACKAGE)): legacy.sha(path) for path in sorted(paths)},
            "uncertainty": {"seed": 20261029, "draws": 10000, "rng": "numpy.PCG64",
                "method": "paired equal-family percentile resampling", "quantile_method": "linear",
                "scope": "Case-bank sensitivity, not population inference"}}

