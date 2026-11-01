"""Fast local checks. Never load a research model or contact a provider."""
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from urllib.parse import unquote, urlsplit
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import notebook_support as nb


class NotebookChecks(unittest.TestCase):
    def test_reviewer_entry_points_and_local_links(self):
        for name in ("README.md", "START_HERE.ipynb", "PREVIEW.html", "scripts/build_notebook.py"):
            self.assertTrue((ROOT/name).is_file(), name)
        for name in ("REPRODUCIBILITY.md", "GPU_SETUP.md", "ANONYMIZATION.md", "SCHEMAS_AND_EXPORTS.md"):
            self.assertTrue((ROOT/"docs"/name).is_file(), name)
            self.assertFalse((ROOT/name).exists(), name)
        self.assertFalse((ROOT/"ARTIFACT_STATUS.md").exists())
        pages=[ROOT/"README.md", *sorted((ROOT/"docs").glob("*.md"))]
        pages.extend(ROOT/"replay"/name/"README.md" for name in
                     ("code", "evidence", "reference_runtime", "generation_inputs"))
        texts=[(page, page.read_text()) for page in pages]
        book=json.loads((ROOT/"START_HERE.ipynb").read_text())
        texts.extend((ROOT/"START_HERE.ipynb", "".join(cell["source"]))
                     for cell in book["cells"] if cell["cell_type"]=="markdown")
        for page,text in texts:
            for target in re.findall(r"\]\(([^)]+)\)",text):
                link=urlsplit(target)
                if link.scheme or link.netloc or not link.path:
                    continue
                resolved=(page.parent/unquote(link.path)).resolve()
                self.assertTrue(resolved.is_relative_to(ROOT), (page.name,target))
                self.assertTrue(resolved.exists(), (page.name,target))

    def test_sources_match(self):
        self.assertGreater(nb.verify_sources()["files_checked"],400)

    def test_cells_parse_and_have_no_errors(self):
        book=json.loads((ROOT/"START_HERE.ipynb").read_text())
        for i,cell in enumerate(book["cells"]):
            if cell["cell_type"]=="code":
                compile("".join(cell["source"]),f"cell {i}","exec")
                self.assertIsNotNone(cell["execution_count"])
                self.assertFalse(any(o["output_type"]=="error" for o in cell["outputs"]))

    def test_all_evaluation_attempts_exported(self):
        with (ROOT/"exports/evaluation_attempts.csv").open() as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),324)
        self.assertEqual(len({r["trial_id"] for r in rows}),324)
        self.assertEqual(sum(r["valid"]=="False" for r in rows),9)

    def test_calibration_exports_keep_failures(self):
        with (ROOT/"exports/calibration_attempts.csv").open() as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),41)
        self.assertEqual(sum(r["valid"]=="False" for r in rows),2)

    def test_four_figures_present(self):
        for name in ("02_decisions_and_agreements","03_payoff_bounds",
                     "04_wording_sensitivity","05_example_payoffs"):
            for suffix in ("png","pdf"):
                self.assertGreater((ROOT/"outputs/figures"/(name+"."+suffix)).stat().st_size,1000)

    def test_invalid_bounds_not_confidence_intervals(self):
        result=nb.read(ROOT/"exports/findings.json")
        primary=next(c for c in result["contrasts"] if c["left"]=="internal" and c["right"]=="unchanged")
        self.assertIsNone(primary["point_estimate"])
        self.assertIsNone(primary["case_bank_sensitivity_interval"])
        self.assertEqual(primary["valid_pairs"],33)
        self.assertAlmostEqual(primary["full_case_bank_invalid_response_bounds"][0],-46.3888888889)
        self.assertAlmostEqual(primary["full_case_bank_invalid_response_bounds"][1],-35.8333333333)

    def test_wording_figure_keeps_missing_values_and_common_scale(self):
        import matplotlib.pyplot as plt
        result=nb.read(ROOT/"exports/findings.json")
        with patch.object(nb,"save",side_effect=lambda fig,name: fig):
            fig=nb.wording_figure(result)
        self.assertEqual(len(fig.axes),3)
        self.assertEqual(len({ax.get_xlim() for ax in fig.axes}),1)
        self.assertEqual(sum(t.get_text()=="invalid" for ax in fig.axes for t in ax.texts),3)
        self.assertEqual(sum(len(ax.collections) for ax in fig.axes),33)
        plt.close(fig)

    def test_count_panels_use_common_scale(self):
        import matplotlib.pyplot as plt
        result=nb.read(ROOT/"exports/findings.json")
        with patch.object(nb,"save",side_effect=lambda fig,name: fig):
            fig=nb.outcome_figure(result)
        self.assertEqual(len({ax.get_xlim() for ax in fig.axes}),1)
        self.assertEqual(len({tuple(ax.get_xticks()) for ax in fig.axes}),1)
        self.assertEqual(len({round(ax.get_position().width,5) for ax in fig.axes}),1)
        plt.close(fig)

    def test_case_figure_identifies_case_and_explains_rule(self):
        import matplotlib.pyplot as plt
        result=nb.read(ROOT/"exports/findings.json")
        case=nb.read(ROOT/"replay/comparison/delivery/cases.json")[0]
        with patch.object(nb,"save",side_effect=lambda fig,name: fig):
            fig=nb.case_figure(result,case["case_id"])
        labels=" ".join(t.get_text() for t in fig.texts)
        self.assertIn(case["case_id"],labels)
        self.assertIn("at least 35 supplier points",labels)
        self.assertIn("Rejection payoffs: buyer 0, supplier 0",labels)
        interpretation=nb.case_interpretation(result,case["case_id"])
        self.assertIn("42 buyer points and 49 supplier points",interpretation)
        self.assertIn("17 supplier points",interpretation)
        self.assertIn("29 supplier points",interpretation)
        plt.close(fig)

    def test_walkthrough_order_and_optional_bounds(self):
        book=json.loads((ROOT/"START_HERE.ipynb").read_text())
        text="\n".join("".join(c["source"]) for c in book["cells"])
        self.assertLess(text.index("nb.candidates(calibration)"),text.index("nb.case_figure"))
        self.assertNotIn("nb.candidate_figure",text)
        self.assertLess(text.index("nb.case_figure"),text.index("nb.outcome_figure"))
        self.assertLess(text.index("nb.outcome_figure"),text.index("nb.wording_figure"))
        self.assertLess(text.index("Optional technical detail: payoff bounds"),text.index("nb.bounds_figure"))
        self.assertLess(text.index("## 6. Export"),text.index("Optional technical detail: payoff bounds"))
        self.assertNotIn("US$",text)  # GitHub treats paired dollar signs as inline math.
        self.assertIn("estimated at USD 11–15",text)

    def test_strength_is_defined_before_selection(self):
        book=json.loads((ROOT/"START_HERE.ipynb").read_text())
        text="\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"]=="markdown")
        self.assertLess(text.index("What does strength mean?"),text.index("2. Compare candidates"))
        self.assertIn("h' = h + av",text)

    def test_selection_table_shows_status_without_redundant_chart(self):
        book=nb.read(ROOT/"START_HERE.ipynb")
        cell=next(c for c in book["cells"] if "selection = nb.candidates(calibration)" in "".join(c["source"]))
        html="".join("".join(o.get("data",{}).get("text/html",[])) for o in cell["outputs"])
        self.assertEqual(html.count("<td>Selected</td>"),1)
        self.assertEqual(html.count("<td>Qualified</td>"),4)
        self.assertEqual(html.count("<td>Failed</td>"),1)
        self.assertIn("Invalid answer",html)
        self.assertFalse(any("image/png" in o.get("data",{}) for o in cell["outputs"]))

    def test_synthetic_task_is_not_evidence(self):
        result=nb.synthetic_example()
        self.assertFalse(result["empirical_evidence"])
        self.assertFalse(result["model_execution"])
        self.assertEqual([r["valid"] for r in result["rows"]],[True,True,False])

    def test_credentials_are_empty(self):
        lines=[line for line in (ROOT/".env.example").read_text().splitlines() if line and not line.startswith("#")]
        self.assertEqual({line.split("=",1)[0] for line in lines},{"MODAL_TOKEN_ID","MODAL_TOKEN_SECRET","HF_TOKEN"})
        self.assertTrue(all(line.split("=",1)[1]=="" for line in lines))
        self.assertIn(".env",(ROOT/".gitignore").read_text().splitlines())

    def test_repeatability_recounts_all_complete_records(self):
        before={p.name:p.read_bytes() for p in (ROOT/"exports").iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            result=nb.repeatability(output_dir=directory)
            self.assertEqual(len(result),2)
            self.assertEqual(result["Behavior"].tolist(),["36/36","36/36"])
            self.assertEqual(result["Edit contrasts"].tolist(),["12/12","12/12"])
            with (Path(directory)/"repeatability_outputs.csv").open() as handle:
                rows=list(csv.DictReader(handle))
            summary=nb.read(Path(directory)/"repeatability_summary.json")
        nb.repeatability()  # The default recount has no export side effects.
        self.assertEqual(before,{p.name:p.read_bytes() for p in (ROOT/"exports").iterdir() if p.is_file()})
        self.assertEqual(len(rows),len({r["trial_id"] for r in rows}))
        self.assertEqual(len(rows),72)
        self.assertFalse(summary["failed_attempt"]["included_in_complete_repeat_totals"])
        text=(ROOT/"START_HERE.ipynb").read_text()
        self.assertIn("Verified identical-prompt repeatability",text)
        self.assertNotIn("Pending verification at notebook preparation",text)

    def test_editable_synthetic_inputs_and_scorer(self):
        result=nb.synthetic_example(budget=130)
        self.assertEqual([r["valid"] for r in result["rows"]],[False,False,True])
        result=nb.synthetic_example(budget=20,responses={"custom":{"team_a":8,"team_b":12}})
        self.assertEqual(result["rows"][0]["outcomes"]["absolute_gap"],4)
        class CustomScorer:
            outcomes=("amount",)
            def score(self,case,raw):
                return {"valid":True,"errors":[],"decision":"custom","outcomes":{"amount":case["budget"]}}
        result=nb.synthetic_example(budget=20,responses={"custom":"authored"},scorer=CustomScorer())
        self.assertEqual(result["rows"][0]["outcomes"],{"amount":20})
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):nb.synthetic_example(budget=invalid)

    def test_construction_pair_recalculates_reference_payoffs(self):
        scenario,frame=nb.construction_example()
        self.assertIn("housings",scenario)
        self.assertEqual(frame["Buyer"].tolist(),[69,84])
        self.assertEqual(frame["Supplier"].tolist(),[52,62])
        self.assertEqual(frame["Combined"].tolist(),[121,146])

    def test_exact_prompt_and_system_addition(self):
        case=nb.read(ROOT/"replay/comparison/delivery/cases.json")[0]
        prompts=nb.input_prompts(case["case_id"])
        self.assertEqual(prompts["unchanged"],prompts["internal"])
        addition=nb.read(ROOT/"replay/protocol/prospective_validation_v2.json")["prompt_text"]
        self.assertEqual(prompts["prompt"][0]["role"],"system")
        self.assertTrue(prompts["prompt"][0]["content"].endswith(addition))
        self.assertEqual(prompts["prompt"][1:],prompts["unchanged"][1:])

    def test_party_means_use_equal_families_on_valid_pairs(self):
        frame=nb.party_payoffs(nb.read(ROOT/"exports/findings.json"))
        edit=frame[frame["Condition"]=="Internal edit"].iloc[0]
        self.assertEqual(edit["Valid pairs"],33)
        self.assertAlmostEqual(edit["Proposed buyer"],11.4166666667)
        self.assertAlmostEqual(edit["Proposed supplier"],-16)
        self.assertAlmostEqual(edit["Buyer after rule"],-19.75)
        self.assertAlmostEqual(edit["Supplier after rule"],-19.25)
        reverse=frame[frame["Condition"]=="Reverse"].iloc[0]
        self.assertAlmostEqual(reverse["Buyer after rule"]+reverse["Supplier after rule"],27.5277777778)

    def test_findings_report_actual_admission(self):
        result=nb.read(ROOT/"exports/findings.json")
        self.assertEqual(result["evidence_label"],"FIXED_COMPARISON_INDEPENDENTLY_VERIFIED")
        self.assertEqual(result["historical_evidence_label"],"SAVED_RESULTS_PENDING_INDEPENDENT_ADMISSION")
        self.assertEqual(result["admission_record"],"replay/comparison/audit.json")

    def test_repeatability_rejects_changed_response(self):
        original_read=nb.read
        def changed(path):
            result=original_read(path)
            if str(path).endswith("combined/outcomes.json"):
                result[0]["raw_response"] += " changed"
            return result
        with patch.object(nb,"read",side_effect=changed),self.assertRaises(AssertionError):
            nb.repeatability()

    def test_repeatability_rejects_duplicate_session(self):
        original_read=nb.read
        def changed(path):
            result=original_read(path)
            if str(path).endswith("combined/summary.json"):
                result["sessions"][1]=result["sessions"][0]
            return result
        with patch.object(nb,"read",side_effect=changed),self.assertRaises(AssertionError):
            nb.repeatability()

    def test_descriptive_prompt_bounds_and_frozen_contrasts(self):
        result=nb.read(ROOT/"exports/descriptive_comparisons.json")
        contrast=result["contrasts"][0]
        self.assertEqual((contrast["left"],contrast["right"]),("prompt","unchanged"))
        self.assertAlmostEqual(contrast["invalid_response_bounds"][0],-35.1111111111)
        self.assertAlmostEqual(contrast["invalid_response_bounds"][1],-17.6944444444)
        self.assertIsNone(contrast["point"])
        self.assertEqual(len(nb.read(ROOT/"exports/findings.json")["contrasts"]),8)

    def test_optimized_mode_rejects_disabled_checks(self):
        for name in ("notebook_support.py","execute_notebook.py","replay/verify_findings.py"):
            result=subprocess.run([sys.executable,"-B","-O",str(ROOT/name)],
                                  capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0,name)
            self.assertIn("required checks must stay enabled",result.stderr,name)

    def test_derivative_transport_inventory(self):
        def digest(data):return hashlib.sha256(data).hexdigest()
        def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        base="replay/generation_inputs/transport.json"
        names=(base,"replay/repeatability/provenance/R01-transport.json",
               "replay/repeatability/provenance/R02-transport.json")
        with zipfile.ZipFile(ROOT/"replay/generation_inputs/source_payload.anonymized.zip") as archive:
            members=set(archive.namelist())
            for name in names:
                transport=nb.read(ROOT/name)
                self.assertEqual(digest(canonical(transport["files"])),transport["inventory_sha256"])
                for row in transport["files"]:
                    member=row["path"]
                    if member in members:data=archive.read(member)
                    elif member=="repeatability/source-transport.json":data=(ROOT/base).read_bytes()
                    elif member=="repeatability/plan.json":
                        path="replay/repeatability/provenance/R01-plan.json" if "R01-" in name else "replay/repeatability/plan.json"
                        data=(ROOT/path).read_bytes()
                    else:
                        session="R01" if "R01-" in name else "R02"
                        data=(ROOT/f"replay/repeatability/source/{session}/{Path(member).name}").read_bytes()
                    self.assertEqual((len(data),digest(data)),(row["size_bytes"],row["sha256"]),(name,member))

    def test_full_cpu_replay_in_scratch_copy(self):
        with tempfile.TemporaryDirectory(prefix="internal-representations-replay-") as directory:
            scratch=Path(directory)
            shutil.copytree(ROOT/"replay",scratch/"replay")
            for name in ("source_manifest.json","START_HERE.ipynb","execute_notebook.py","notebook_support.py"):
                shutil.copy2(ROOT/name,scratch/name)
            (scratch/"verification").mkdir()
            # Copy any separately inventoried input, without unrelated outputs or Git metadata.
            for name in nb.read(ROOT/"source_manifest.json")["sha256"]:
                target=scratch/name
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(ROOT/name,target)
            result=subprocess.run([sys.executable,"-B",str(scratch/"execute_notebook.py")],
                cwd=scratch,env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
                capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-2000:])
            self.assertEqual(nb.read(scratch/"verification/notebook_execution.json")["status"],"PASS")
            actual=nb.read(scratch/"exports/findings.json")
            expected=nb.read(ROOT/"exports/findings.json")
            for key in expected:
                if key!="analysis_provenance":self.assertEqual(actual[key],expected[key],key)
            for name in ("evaluation_attempts.csv","calibration_attempts.csv","repeatability_outputs.csv",
                         "party_payoff_differences.csv"):
                self.assertEqual((scratch/"exports"/name).read_bytes(),(ROOT/"exports"/name).read_bytes(),name)
            self.assertEqual(nb.read(scratch/"exports/descriptive_comparisons.json")["contrasts"],
                             nb.read(ROOT/"exports/descriptive_comparisons.json")["contrasts"])


if __name__=="__main__":unittest.main()
