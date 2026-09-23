"""Tests for the procurement extraction prototype.

Run: python -m unittest test_pipeline -v -b
The lecture categories organise the tests. Model accuracy is evaluated separately.
Integration checks cover mixed PDF reading and saving temporary files.
Framework: https://docs.python.org/3/library/unittest.html
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import fitz

from langchain_core.messages import AIMessage, ToolMessage
import extract_graph as pipe
import readers
import review_standards as review
import standards
import validate_extraction as val
from criteria_workers import collect_findings, prepare_finding, Finding, make_plan, correct_once


def checked_criteria(items):
    finding = {"criteria": [{"lot": "1", **c} for c in items], "weight_scale": "points",
               "unresolved_lots": [], "scope_unclear": False,
               "source_conflict": False, "evidence_incomplete": False}
    return collect_findings([{ "task": {"lot_labels": ["1"]},
                              "status": "completed", "finding": finding}],
                            {"division": "not_divided"})


class Test_interface(unittest.TestCase):
    def test_reference_parser_returns_id_and_fields(self):
        record = {"tenderingProcess": {"documentsURL": "/Procurement/125861"},
                  "cpvType": "71220000-6"}
        eis_id, fields = val.simplify_ground_truth(record)
        self.assertEqual(eis_id, "125861")
        self.assertEqual(fields, {"main_cpv": "71220000-6",
                                  "additional_cpv": [], "criteria": [], "criteria_lots": []})
        self.assertIsNone(val.simplify_ground_truth({})[0])

    def test_saved_record_contract(self):
        state = {"final": {"main_cpv": "71220000-6"},
                 "cpv_seconds": 2.0, "criteria_seconds": 1.0}
        record = pipe.build_record("125861", state)
        self.assertEqual(record["eis_id"], "125861")
        self.assertEqual(record["extracted"], state["final"])
        self.assertEqual(record["model_seconds"], 3.0)
        self.assertEqual(set(record), {
            "eis_id", "source_files", "candidates", "attempts", "model_seconds",
            "extracted", "critique", "evaluation_criteria", "criteria_check", "standards", "lots", "lot_seconds"})


class Test_data_structures(unittest.TestCase):
    def test_duplicate_names_merge_but_distinct_roles_remain(self):
        items = [{"name": "Cena", "weight": 50}, {"name": " CENA ", "weight": 50},
                 {"name": "Experience (architect)", "weight": 25},
                 {"name": "Experience (engineer)", "weight": 25}]
        f = Finding(criteria=[{"lot":"1", "evidence_ids":["a"], "scope_evidence":"Heading", **c} for c in items],
                    weight_scale="points", unresolved_lots=[], scope_unclear=False,
                    source_conflict=False, evidence_incomplete=False)
        result, _ = prepare_finding(f, "1", {"division":"not_divided"})
        self.assertEqual([c.name for c in result.criteria],
                         ["Cena", "Experience (architect)", "Experience (engineer)"])

    def test_multiset_scoring_keeps_repeated_weights(self):
        counts = {"tp": 0, "fp": 0, "fn": 0}
        val.score_multiset([30, 30, 40], [30, 40, 40], counts)
        self.assertEqual(counts, {"tp": 2, "fp": 1, "fn": 1})

    def test_cpv_in_a_table_is_a_candidate(self):
        state = {"documents_text": "No code in prose.",
                 "tables": [{"rows": [["CPV", "71220000-6"]]}]}
        result = pipe.find_candidates(state)
        self.assertEqual([c["code"] for c in result["candidates"]], ["71220000-6"])


    def test_evidence_sample_keeps_first_and_last_mentions(self):
        occurrences = [("selection", f"Passage {i}") for i in range(7)]
        with patch.object(standards, "MAX_EVIDENCE", 4):
            result = standards.sample_evidence(occurrences)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["text"], "Passage 0")
        self.assertEqual(result[-1]["text"], "Passage 6")

    def test_merged_table_cells_keep_column_alignment(self):
        html = ('<table><tr><td rowspan="2" colspan="2">Group</td><td>A</td></tr>'
                '<tr><td>B</td></tr></table>')
        self.assertEqual(readers._html_to_rows(html), [["Group", "", "A"], ["", "", "B"]])

    def test_final_human_amendment_overrides_disagreeing_receipt(self):
        original = {"name": "ISO 9001", "applies": True, "reason": "Proposed"}
        edited = {**original, "applies": False, "reason": "Human amended"}
        messages = [AIMessage(content="", tool_calls=[
            {"name": "request_review", "id": "1", "args": original}]),
            ToolMessage(content=json.dumps({**original, "decision_recorded": True}), tool_call_id="1")]
        decision = {"type": "edit", "edited_action": {"name": "request_review", "args": edited}}
        result = review.review_outcome(SimpleNamespace(value={"messages": messages}),
                                       "ISO 9001", [decision], 0.1, 1, True)
        self.assertFalse(result["applies"])
        self.assertEqual(result["reason"], "Human amended")
        self.assertEqual(result["decision_source"], "human_edit_reconciled")
        self.assertIn("amend_discrepancy", result)


class Test_boundary_conditions(unittest.TestCase):
    def test_cpv_shape(self):
        # Shape only: this does not check membership in the official CPV catalogue.
        for code, expected in [("71000000-8", True), ("71000", False), ("", False)]:
            with self.subTest(code=code):
                self.assertEqual(pipe.looks_like_cpv(code), expected)

    def test_sole_criterion_default_preserves_explicit_weight(self):
        for weight, expected in [(None, 100), (50, 50)]:
            with self.subTest(weight=weight):
                result = checked_criteria([{"name": "Price", "weight": weight}])
                self.assertEqual(result["criteria"][0]["weight"], expected)
                self.assertEqual(result["criteria"][0]["raw_weight"], weight)
                self.assertEqual(result["weights_reconcile"], expected == 100)

    def test_reference_total_before_rounding(self):
        for values, expected in [([], "missing"), ([None], "incomplete"),
                                 ([1], "total_mismatch"), ([100, 100, 100], "total_mismatch"),
                                 ([33.33, 33.33, 33.33], "available"),
                                 ([99.5], "available"), ([99.49], "total_mismatch")]:
            with self.subTest(values=values):
                criteria = [{"weight": w} for w in values]
                self.assertEqual(val.criteria_reference(criteria)["status"], expected)

    def test_weight_parser_keeps_the_integer_policy(self):
        for raw, expected in [(None, []), ("30,5", [30]), (31.5, [32]),
                              ("50 points out of 100", [50]), ("-10", [])]:
            with self.subTest(raw=raw):
                self.assertEqual(val.weight_values(raw), expected)

    def test_empty_metrics_do_not_divide_by_zero(self):
        self.assertEqual(val.prf({"tp": 0, "fp": 0, "fn": 0}), (0.0, 0.0, 0.0))

    def test_short_text_is_not_discarded(self):
        text, tables = readers.read_file(b"CPV 71220000-6", "notice.txt")
        self.assertIn("71220000-6", text)
        self.assertEqual(tables, [])


    def test_separate_lots_do_not_share_a_single_total(self):
        finding = {"criteria": [{"lot":"1", "name":"Cena", "weight":100}],
                   "weight_scale":"points", "unresolved_lots":[], "scope_unclear":False,
                   "source_conflict":False, "evidence_incomplete":False}
        records = [{"task":{"lot_labels":[label]}, "status":"completed",
                    "finding": {**finding, "criteria":[{"lot":label,"name":"Cena","weight":100}]}}
                   for label in ["1", "2"]]
        result = collect_findings(records, {"labels_complete":True, "count":2})
        self.assertTrue(result["weights_reconcile"])
        self.assertEqual([c["total"] for c in result["lot_checks"].values()], [100,100])

    def test_decimal_weights_reconcile_without_integer_rounding(self):
        result = checked_criteria([{"name": n, "weight": 33.33} for n in ("Price","Quality","Delivery")])
        self.assertTrue(result["weights_reconcile"])
        self.assertAlmostEqual(result["lot_checks"]["1"]["total"], 99.99)


class Test_execution_paths(unittest.TestCase):
    def test_scalar_scoring_branches(self):
        cases = [("A", "A", (1, 0, 0)), ("B", "A", (0, 1, 1)),
                 ("", "A", (0, 0, 1)), ("A", "", (0, 1, 0)), ("", "", (0, 0, 0))]
        for prediction, reference, expected in cases:
            with self.subTest(prediction=prediction, reference=reference):
                counts = {"tp": 0, "fp": 0, "fn": 0}
                val.score_scalar(prediction, reference, counts)
                self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), expected)

    def test_criteria_reconcile_or_remain_unresolved(self):
        for values, expected in [([60,40],True),([30,40],False),([100,None],False)]:
            with self.subTest(values=values):
                result = checked_criteria([{"name":n, "weight":w} for n,w in zip(["Price","Quality"],values)])
                self.assertEqual(result["weights_reconcile"], expected)

    def test_cpv_final_status_preserves_provisional_codes(self):
        for verdict, expected in [("accept", "accepted"), ("revise", "unresolved")]:
            with self.subTest(verdict=verdict):
                result = pipe.finalize({
                    "classification": {"main_cpv": "71220000-6", "additional_cpv": []},
                    "critique": {"verdict": verdict}, "attempts": 3})["final"]
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["main_cpv"], "71220000-6")
                self.assertEqual(result["retry_exhausted"], verdict == "revise")

    def test_evaluation_separates_missing_output_and_missing_reference(self):
        gold = {"main_cpv": "71220000-6", "additional_cpv": [], "criteria": []}
        extracted = {**gold, "status": "unresolved", "criteria": [{"weight": 100}]}
        rows, summary = val.evaluate({"1": gold, "2": gold}, {"1": extracted})
        self.assertEqual(summary["matched_records"]["main_cpv"]["exact_match_rate"], 1.0)
        self.assertNotIn("reference_universe", summary)
        self.assertEqual(summary["matched_records"]["criteria"]["records"], 0)
        self.assertEqual(summary["selected_ids"], ["1"])
        self.assertEqual([row["eis_id"] for row in rows], ["1"])
        self.assertEqual(summary["cpv_status_strata"]["unresolved"]["outputs"], 1)

    def test_legal_patterns_preserve_type_and_sentence_boundary(self):
        cases = [
            ("Direktīvu 95/46/EK", ["EU directive 95/46"]),
            ("Ministru kabineta 2013. gada 8. oktobra noteikumi Nr. 1041",
             ["Ministru kabineta noteikumi Nr. 1041"]),
            ("Ministru kabineta kompetencē. Pašvaldības noteikumi Nr. 5", [])]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual([f["name"] for f in standards.find_standards(text)], expected)


    def test_count_alone_does_not_invent_lot_labels(self):
        self.assertEqual(len(make_plan({"labels":["2","4"],"count":6})["tasks"]), 2)
        plan = make_plan({"labels": [], "count": 6})
        self.assertEqual(plan["tasks"][0]["lot_labels"], [None])
        self.assertTrue(plan["tasks"][0]["fallback"])

    def test_review_summary_distinguishes_attempts_from_decisions(self):
        findings = [{"review": {"status": "decided", "applies": True}},
                    {"review": {"status": "unresolved", "applies": None,
                                "human": [{"type": "reject"}]}}, {}]
        result = review.review_summary(findings, "test model")
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["decided"], 1)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["not_attempted"], 1)
        self.assertEqual(result["human_reviewed"], 1)
        self.assertFalse(result["complete"])


class Test_error_handling(unittest.TestCase):
    def test_malformed_model_code_does_not_reject_valid_codes(self):
        # Regression inspired by the reported procurement 162492 failure.
        result = pipe.CpvClassification(main_cpv="71000", reasoning="Example",
                                        additional_cpv=["71000", "71220000-6"])
        self.assertEqual(result.main_cpv, "")
        self.assertEqual(result.additional_cpv, ["71220000-6"])

    def test_classifier_failure_records_the_error(self):
        model = Mock()
        model.invoke.side_effect = RuntimeError("Service unavailable")
        state = {"attempts": 0, "candidates": [{"code": "71220000-6", "context": "Main CPV"}]}
        with patch.object(pipe, "classifier", model):
            result = pipe.classify(state)
        model.invoke.assert_called_once()
        self.assertIsNone(result["classification"])
        self.assertEqual(result["cpv_error"]["stage"], "classify")
        self.assertEqual(result["cpv_error"]["message"], "Service unavailable")

    def test_review_needs_a_matching_successful_receipt(self):
        proposed = {"name": "ISO 9001", "applies": True, "reason": "Proposed"}
        receipt = {**proposed, "applies": False, "reason": "Amended", "decision_recorded": True}
        for reply_id, status in [(None, None), ("wrong", "success"), ("1", "error"), ("1", "success")]:
            with self.subTest(reply_id=reply_id, status=status):
                messages = [AIMessage(content="", tool_calls=[
                    {"name": "request_review", "id": "1", "args": proposed}])]
                if reply_id is not None:
                    messages.append(ToolMessage(content=json.dumps(receipt), tool_call_id=reply_id, status=status))
                result = review.last_tool_call(SimpleNamespace(value={"messages": messages}), "ISO 9001")
                if reply_id == "1" and status == "success":
                    self.assertFalse(result["applies"])
                    self.assertEqual(result["reason"], "Amended")
                else:
                    self.assertIsNone(result)

    def test_duplicate_reference_error_identifies_the_record(self):
        record = {"tenderingProcess": {"documentsURL": "/Procurement/123"}}
        with patch.object(Path, "read_text", return_value=json.dumps([record, record])):
            with self.assertRaisesRegex(ValueError, "Duplicate reference ID 123.*1 and 2"):
                val.load_ground_truth("unused.json")


    def test_critic_failure_keeps_the_provisional_code(self):
        state = {"classification": {"main_cpv": "71220000-6", "additional_cpv": [], "reasoning": "Proposed"},
                 "candidates": [{"code": "71220000-6", "context": "Main CPV"}], "attempts": 3}
        model = Mock()
        model.invoke.side_effect = RuntimeError("Critic unavailable")
        with patch.object(pipe, "critic", model):
            state.update(pipe.critique(state))
        result = pipe.finalize(state)["final"]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["main_cpv"], "71220000-6")
        self.assertEqual(result["error"]["message"], "Critic unavailable")

    def test_invalid_human_response_is_reprompted(self):
        finding = {"name": "ISO 9001", "category": "standard", "phases": [], "count": 1, "evidence": []}
        request = {"args": {"applies": True, "reason": "Proposed"}}
        with patch("builtins.input", side_effect=["maybe", "a"]) as answer:
            decision = review.ask_person(finding, request)
        self.assertEqual(answer.call_count, 2)
        self.assertEqual(decision["type"], "approve")


class Test_metamorphic(unittest.TestCase):
    def test_reordering_weights_does_not_change_the_score(self):
        prediction, reference = [30, 30, 40], [30, 40, 40]
        original = {"tp": 0, "fp": 0, "fn": 0}
        reordered = {"tp": 0, "fp": 0, "fn": 0}
        val.score_multiset(prediction, reference, original)
        val.score_multiset(list(reversed(prediction)), list(reversed(reference)), reordered)
        self.assertEqual(original, reordered)


class Test_criteria_correction(unittest.TestCase):
    """Numerical feedback corrects omissions without losing a valid first answer."""

    def record(self):
        item = {"lot": "2", "name": "Enerģija", "weight": 1,
                "evidence_ids": ["p1"], "scope_evidence": "2. daļa"}
        finding = {"criteria": [item], "weight_scale": "points", "unresolved_lots": [],
                   "scope_unclear": False, "source_conflict": False, "evidence_incomplete": False}
        return {"task": {"lot_labels": ["2"]}, "status": "completed",
                "finding": finding, "normalizations": []}

    def test_numerical_feedback_recovers_omitted_criteria(self):
        record = self.record()
        original = record["finding"]
        corrected = Finding(**{**original, "criteria": [
            {**original["criteria"][0], "name": name, "weight": weight}
            for name, weight in [("Cena", 98.5), ("Enerģija", 1), ("Tehniskie punkti", 0.5)]]})
        call = Mock(return_value=corrected)
        lots = {"labels": ["2"], "labels_complete": True}
        correct_once(call, "worker_1_correction", "Original evidence", record, [{"id": "p1"}], lots)
        call.assert_called_once()
        self.assertTrue(call.call_args.args[2].startswith("Original evidence"))
        self.assertIn("returned total=1", call.call_args.args[2])
        self.assertTrue(collect_findings([record], lots)["weights_reconcile"])
        self.assertEqual(len(record["finding"]["criteria"]), 3)

    def test_failed_correction_preserves_original(self):
        record = self.record()
        original = json.loads(json.dumps(record["finding"]))
        call = Mock(side_effect=RuntimeError("Model unavailable"))
        correct_once(call, "worker_1_correction", "Original evidence", record, [{"id": "p1"}], {})
        call.assert_called_once()
        self.assertEqual(record["finding"], original)
        self.assertEqual(record["correction"]["status"], "error")
        self.assertEqual(collect_findings([record], {})["lot_checks"]["2"]["status"], "total_mismatch")


class Test_integration(unittest.TestCase):
    """Component integration checks with real PDF parsing and temporary JSON files."""
    def test_save_result_writes_record_tables_and_ocr(self):
        state = {"final": {"main_cpv": "71220000-6"},
                 "table_records": [{"name": "table"}], "ocr_records": [{"name": "scan"}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, tables, ocr = pipe.save_result(
                "125861", state, root / "extracted", root / "tables", root / "ocr")
            self.assertEqual(json.loads(output.read_text())["extracted"], state["final"])
            self.assertEqual(json.loads(tables.read_text()), state["table_records"])
            self.assertEqual(json.loads(ocr.read_text()), state["ocr_records"])


    def test_mixed_pdf_keeps_digital_and_scanned_pages(self):
        with fitz.open() as scan:
            page = scan.new_page()
            page.insert_text((72, 72), "Scanned criterion")
            image = page.get_pixmap().tobytes("png")
        with fitz.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "DIGITAL_MARKER " + "Tender information. " * 8)
            page = document.new_page()
            page.insert_image(page.rect, stream=image)
            data = document.tobytes()
        info = {}
        with patch.object(readers, "ENABLE_TABLE_TRANSFORMER", False), \
                patch.object(readers, "read_ocr", return_value="SCANNED_MARKER quality 40") as ocr:
            text, _ = readers.read_file(data, "mixed.pdf", info)
        ocr.assert_called_once()
        self.assertLess(text.index("DIGITAL_MARKER"), text.index("SCANNED_MARKER"))
        self.assertEqual(info["ocr_page_numbers"], [2])
        self.assertEqual(info["ocr_text"], "SCANNED_MARKER quality 40")


if __name__ == "__main__":
    unittest.main()
