"""Fast local checks. Never load a research model or contact a provider."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import notebook_support as nb


class NotebookChecks(unittest.TestCase):
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

    def test_five_figures_present(self):
        for name in ("01_candidate_selection","02_decisions_and_agreements","03_payoff_bounds",
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
        self.assertLess(text.index("nb.candidate_figure"),text.index("nb.case_figure"))
        self.assertLess(text.index("nb.case_figure"),text.index("nb.outcome_figure"))
        self.assertLess(text.index("nb.outcome_figure"),text.index("nb.wording_figure"))
        self.assertLess(text.index("Optional technical detail: payoff bounds"),text.index("nb.bounds_figure"))
        self.assertLess(text.index("## 6. Export"),text.index("Optional technical detail: payoff bounds"))

    def test_strength_is_defined_before_selection(self):
        book=json.loads((ROOT/"START_HERE.ipynb").read_text())
        text="\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"]=="markdown")
        self.assertLess(text.index("What does strength mean?"),text.index("2. Compare candidates"))
        self.assertIn("h' = h + av",text)

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


if __name__=="__main__":unittest.main()
