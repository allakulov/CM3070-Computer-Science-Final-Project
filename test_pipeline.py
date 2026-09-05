"""Unit tests for the deterministic parts of the pipeline.

Test the functions that have a fixed input and a fixed output: the CPV code
checks, the criterion handling, the weight sum, and the validator scoring.
The language model is not tested, models outputs are evaluated separately. 
The tests focus on edge cases and on bugs encountered during prototype evaluation runs.

Run:
    python -m unittest test_pipeline -v
"""

import unittest

import extract_graph as pipe
import validate_extraction as val


# INTERFACE: the function is called correctly and returns what is expected.

class TestInterface(unittest.TestCase):
    def test_simplify_ground_truth_returns_id_and_fields(self):
        eis_id, fields = val.simplify_ground_truth(
            {"tenderingProcess": {"documentsURL": ".../Procurement/125861"}})
        self.assertEqual(eis_id, "125861")
        self.assertIn("main_cpv", fields)


# DATA STRUCTURES: lists and dictionaries hold the right contents.

class TestDataStructures(unittest.TestCase):
    def test_dedup_criteria_keeps_first_of_each_name(self):
        items = [{"name": "Cena", "weight": 60}, {"name": "cena", "weight": 40}]
        deduped = pipe.dedup_criteria(items)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["weight"], 60)

    def test_score_multiset_counts_shared_extra_missing(self):
        counts = {"tp": 0, "fp": 0, "fn": 0}
        val.score_multiset([30, 30, 40], [30, 40, 40], counts)
        self.assertEqual(counts, {"tp": 2, "fp": 1, "fn": 1})

    def test_weights_returns_sorted_multiset(self):
        self.assertEqual(val.weights([{"weight": 40}, {"weight": 30}, {"weight": 30}]),
                         [30, 30, 40])


# BOUNDARY CONDITIONS: empty inputs, a single item, and values at the edge.

class TestBoundaryConditions(unittest.TestCase):
    def test_cpv_form_at_the_edges(self):
        self.assertTrue(pipe.looks_like_cpv("71000000-8"))   # a valid CPV code
        self.assertFalse(pipe.looks_like_cpv("71000"))       # too short (the 162492 case)
        self.assertFalse(pipe.looks_like_cpv(""))            # empty string

    def test_weight_values_empty_and_single(self):
        self.assertEqual(val.weight_values(None), [])
        self.assertEqual(val.weight_values(""), [])
        self.assertEqual(val.weight_values(30), [30])

    def test_single_criterion_gets_full_weight(self):
        state = {"criteria": {"criteria": [{"name": "Cena", "weight": None}]}}
        out = pipe.check_criteria(state)
        self.assertEqual(out["criteria"]["criteria"][0]["weight"], 100)
        self.assertEqual(out["criteria_check"], "ok")

    def test_prf_on_empty_counts_does_not_divide_by_zero(self):
        self.assertEqual(val.prf({"tp": 0, "fp": 0, "fn": 0}), (0.0, 0.0, 0.0))


# EXECUTION PATHS: every branch of the function is run.

class TestExecutionPaths(unittest.TestCase):
    def test_score_scalar_all_four_paths(self):
        counts = {"tp": 0, "fp": 0, "fn": 0}
        val.score_scalar("45200000-9", "45200000-9", counts)      # correct code
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 0, 0))
        val.score_scalar("11111111-1", "45200000-9", counts)      # wrong code: fp and fn
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 1, 1))
        val.score_scalar("", "45200000-9", counts)                # nothing extracted: fn
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 1, 2))
        val.score_scalar("45200000-9", "", counts)                # truth empty: fp
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 2, 2))

    def test_find_total_points_found_versus_default(self):
        stated = {"documents_text": "Maksimālais iespējamais punktu skaits: 60"}
        self.assertEqual(pipe.find_total_points(stated), 60.0)
        self.assertEqual(pipe.find_total_points({"documents_text": "no total here"}), 100.0)

    def test_check_criteria_reconciled_revise_and_no_weights(self):
        ok = pipe.check_criteria({"criteria": {"criteria": [
            {"name": "a", "weight": 60}, {"name": "b", "weight": 40}]}})
        self.assertEqual(ok["criteria_check"], "ok")
        self.assertTrue(ok["criteria"]["weights_reconcile"])

        revise = pipe.check_criteria({"criteria_attempts": 0, "criteria": {"criteria": [
            {"name": "a", "weight": 30}, {"name": "b", "weight": 40}]}})
        self.assertEqual(revise["criteria_check"], "revise")

        none = pipe.check_criteria({"criteria": {"criteria": [
            {"name": "a", "weight": None}, {"name": "b", "weight": None}]}})
        self.assertEqual(none["criteria_check"], "ok")


# ERROR HANDLING: malformed input is handled instead of crashing (the 162492 case).

class TestErrorHandling(unittest.TestCase):
    def test_malformed_additional_code_is_dropped_not_raised(self):
        # The 162492 case: one too-short code among valid ones. It is dropped and the
        # record still parses, instead of the whole classification failing.
        result = pipe.CpvClassification(
            main_cpv="45200000-9", reasoning="x",
            additional_cpv=["71000", "71000000-8", "71248000-8", "71242000-6"])
        self.assertNotIn("71000", result.additional_cpv)
        self.assertEqual(result.additional_cpv, ["71000000-8", "71248000-8", "71242000-6"])

    def test_malformed_main_code_becomes_empty_not_raised(self):
        result = pipe.CpvClassification(main_cpv="71000", reasoning="x", additional_cpv=[])
        self.assertEqual(result.main_cpv, "")

    def test_ground_truth_without_url_yields_no_id(self):
        eis_id, _ = val.simplify_ground_truth({"tenderingProcess": {}})
        self.assertIsNone(eis_id)


# AFTER REFACTORING: the two shared functions that the CLI and the app both call.

class TestSaveExtraction(unittest.TestCase):
    """Test build_record and save_result.

    build_record does the record for one procurement. save_result writes the record
    and its tables and OCR to three different files. The command line and streamlit app both 
    call the two functions, so the record must hold the right fields and the files must be written
    to the right folders.
    """

    def sample_state(self):
        """A run state with one value for every field the two functions read."""
        return {"source_files": ["a.pdf"], "candidates": [{"code": "45000000-7"}],
                "attempts": 1, "cpv_seconds": 2.0, "criteria_seconds": 1.0,
                "final": {"main_cpv": "45000000-7"}, "critique": {"verdict": "accept"},
                "criteria": {"found": True, "criteria": []}, "criteria_check": "ok",
                "standards": {}, "table_records": [{"name": "t"}], "ocr_records": [{"name": "o"}]}

    def test_record_has_the_expected_fields(self):
        # The record must hold these ten fields and no others. model_seconds is the sum
        # of the two model times (2.0 + 1.0 = 3.0), the one value the record computes.
        record = pipe.build_record("125861", self.sample_state())
        self.assertEqual(record["eis_id"], "125861")
        self.assertEqual(record["model_seconds"], 3.0)
        self.assertEqual(record["extracted"], {"main_cpv": "45000000-7"})
        self.assertEqual(set(record), {"eis_id", "source_files", "candidates", "attempts",
                                       "model_seconds", "extracted", "critique",
                                       "evaluation_criteria", "criteria_check", "standards"})

    def test_save_result_writes_three_files(self):
        # save_result writes three files in the folders: the record to the first
        # the tables to the second, the OCR to the third. It also creates the
        # folders, so a temporary directory that does not exist yet suffices.
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out, tab, ocr = pipe.save_result("125861", self.sample_state(),
                                             Path(tmp) / "extracted", Path(tmp) / "tables", Path(tmp) / "ocr")
            self.assertEqual(json.loads(out.read_text())["eis_id"], "125861")
            self.assertEqual(json.loads(tab.read_text()), [{"name": "t"}])
            self.assertEqual(json.loads(ocr.read_text()), [{"name": "o"}])


if __name__ == "__main__":
    unittest.main(argv=["ignored", "-v"], exit=False)