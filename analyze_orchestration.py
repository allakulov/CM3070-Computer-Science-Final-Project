"""Summarize saved criteria corrections and CPV review decisions.

Run: python analyze_orchestration.py RUN_FOLDER
Uses only the standard library and files already saved in the run folder.
Numerical reconciliation is reported separately from reference agreement.
"""

import argparse
import csv
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_cpv_log(path):
    """Keep classification attempts and critic requests under their procurement."""
    records = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Procurement "):
            current = line.split()[1]
            if current in records:
                raise ValueError(f"Repeated procurement {current} in terminal log.")
            records[current] = {"answers": [], "revisions": 0}
        elif current:
            match = re.search(r"classify \(attempt (\d+)\): main (\S+), additional (\[[^\]]*\])", line)
            if match:
                records[current]["answers"].append({
                    "attempt": int(match[1]),
                    "main": "" if match[2] == "None" else match[2],
                    "additional": re.findall(r"\d{8}-\d", match[3]),
                })
            if "critique: revise" in line:
                records[current]["revisions"] += 1
    return records


def codes(value):
    return set(re.findall(r"\d{8}-\d", value))


def set_f1(predicted, reference):
    return 2 * len(predicted & reference) / (len(predicted) + len(reference))


def change(before, after):
    if after > before:
        return "improved"
    if after < before:
        return "worsened"
    return "unchanged"


def cpv_row(record, reference, log):
    """Compare revisions only where the reference contains a value."""
    result = record["extracted"]
    final_main = result.get("main_cpv") or ""
    final_additional = set(result.get("additional_cpv", []))
    first = next((answer for answer in log["answers"] if answer["attempt"] == 1), None)
    if log["answers"]:
        last = log["answers"][-1]
        if last["main"] != final_main or set(last["additional"]) != final_additional:
            raise ValueError(f"Terminal log and saved CPV output disagree for {record['eis_id']}.")
    if reference["main_cpv_extracted"] != final_main or codes(reference["additional_cpv_extracted"]) != final_additional:
        raise ValueError(f"Evaluation CSV and saved CPV output disagree for {record['eis_id']}.")
    main_truth = reference["main_cpv_truth"].strip()
    additional_truth = codes(reference["additional_cpv_truth"])
    main_before = first["main"] == main_truth if first and main_truth else ""
    main_after = final_main == main_truth if main_truth else ""
    additional_before = set_f1(set(first["additional"]), additional_truth) if first and additional_truth else ""
    additional_after = set_f1(final_additional, additional_truth) if additional_truth else ""
    return {
        "procurement_id": record["eis_id"],
        "classification_attempts": record["attempts"],
        "critic_revision_requests": log["revisions"],
        "final_status": result["status"],
        "retry_exhausted": result.get("retry_exhausted", ""),
        "main_before": first["main"] if first else "",
        "main_after": final_main,
        "main_reference": main_truth,
        "main_match_before": main_before,
        "main_match_after": main_after,
        "main_agreement_change": change(main_before, main_after) if main_before != "" else "not compared",
        "additional_before": "; ".join(sorted(first["additional"])) if first else "",
        "additional_after": "; ".join(sorted(final_additional)),
        "additional_reference": "; ".join(sorted(additional_truth)),
        "additional_f1_before": additional_before,
        "additional_f1_after": additional_after,
        "additional_agreement_change": change(additional_before, additional_after) if additional_before != "" else "not compared",
        "reference_note": "" if additional_truth else "Empty additional-code reference; check additions as recovery.",
    }


def criteria_rows(record, workers, calls):
    """Use stored checks; do not reconstruct the extraction's validation rules."""
    rows = []
    for number, worker in enumerate(workers, 1):
        correction = worker.get("correction", {})
        after = correction.get("after_check", {})
        stage = f"worker_{number}_correction"
        correction_calls = [call for call in calls if call["stage"] == stage]
        reconciled = bool(after) and all(check["status"] == "reconciled" for check in after.values())
        labels = worker["task"]["lot_labels"]
        default_used = any(entry.get("weight_source") == "sole_criterion_default"
                           and entry.get("lot") in labels
                           for entry in record["evaluation_criteria"]["criteria"])
        rows.append({
            "procurement_id": record["eis_id"],
            "worker": number,
            "lot": "; ".join(str(label) for label in worker["task"]["lot_labels"] if label is not None) or "unknown",
            "worker_status": worker["status"],
            "correction_attempted": bool(correction),
            "correction_status": correction.get("status", "not requested"),
            "checks_before": json.dumps(correction.get("before_check", {}), ensure_ascii=False),
            "checks_after": json.dumps(after, ensure_ascii=False),
            "reconciled_after_correction": reconciled if correction else "",
            "sole_weight_default_after_correction": default_used if correction else "",
            "added_passages": len(correction.get("added_passage_ids", [])),
            "correction_calls": len(correction_calls),
            "output_limit_retries": sum(call["attempt"] > 1 for call in correction_calls),
            "correction_seconds": round(sum(call["seconds"] for call in correction_calls), 2),
            "error": correction.get("error", ""),
        })
    return rows


def summarize(workers, cpv, model_seconds):
    corrected = [row for row in workers if row["correction_attempted"]]
    reconciled = [row for row in corrected if row["reconciled_after_correction"]]
    revised = [row for row in cpv if row["critic_revision_requests"]]
    wrong = [row for row in cpv if row["main_reference"] and row["main_after"] and not row["main_match_after"]]
    correction_seconds = sum(row["correction_seconds"] for row in workers)
    metrics = [
        ("Procurements analyzed", len(cpv)),
        ("Criteria workers", len(workers)),
        ("Workers receiving correction", f"{len(corrected)}/{len(workers)}"),
        ("Procurements receiving criteria correction", len({row['procurement_id'] for row in corrected})),
        ("Corrected workers ending reconciled", f"{len(reconciled)}/{len(corrected)}"),
        ("Reconciled corrections using a sole-weight default", sum(row["sole_weight_default_after_correction"] for row in reconciled)),
        ("Workers whose correction failed after retries", sum(row["correction_status"] == "error" for row in corrected)),
        ("Criteria correction model calls", sum(row["correction_calls"] for row in workers)),
        ("Additional calls caused by correction output limits", sum(row["output_limit_retries"] for row in workers)),
        ("Criteria correction seconds", f"{correction_seconds:.2f}"),
        ("Total criteria model-call seconds", f"{model_seconds:.2f}"),
        ("Procurements with a CPV critic revision request", len(revised)),
        ("Revised main codes: agreement improved", sum(row["main_agreement_change"] == "improved" for row in revised)),
        ("Revised main codes: agreement worsened", sum(row["main_agreement_change"] == "worsened" for row in revised)),
        ("Revised main codes: agreement unchanged", sum(row["main_agreement_change"] == "unchanged" for row in revised)),
        ("Revised additional codes with eligible references", sum(bool(row["additional_reference"]) for row in revised)),
        ("Accepted CPV outputs", sum(row["final_status"] == "accepted" for row in cpv)),
        ("CPV outputs with no candidates", sum(row["final_status"] == "no_candidates" for row in cpv)),
        ("Nonmatching main codes marked accepted", f"{sum(row['final_status'] == 'accepted' for row in wrong)}/{len(wrong)}"),
        ("CPV outputs with retries exhausted", sum(row["retry_exhausted"] is True for row in cpv)),
    ]
    table = "| Measure | Result |\n| --- | --- |\n"
    table += "\n".join(f"| {name} | {value} |" for name, value in metrics)
    return table + "\n\n" + (
        "Reconciliation means the stored check passed; it does not establish correctness. "
        "The checks include the sole-criterion weight default. Workers include fallback tasks "
        "and procurements with unavailable criteria references.\n\n"
        "CPV agreement uses the reference values already recorded in evaluation.csv. "
        "Empty references are not scored. Additions to empty lists need document-based recovery review. "
        "The CPV CSV includes all outputs so accepted errors and missing outputs remain visible.\n\n"
        "Times are recorded model-call durations, including failed calls and retries. "
        "They are not total extraction wall time. A correction changes both feedback and evidence, "
        "so this analysis cannot isolate the benefit of either. Criterion-name quality is not scored here.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run folder containing extracted, traces, evaluation.csv and the terminal log.")
    parser.add_argument("--output", type=Path, help="Output folder; default: RUN_FOLDER/orchestration_audit.")
    args = parser.parse_args()
    try:
        with (args.run / "evaluation.csv").open(encoding="utf-8-sig", newline="") as file:
            references = {}
            for row in csv.DictReader(file):
                if row["eis_id"] in references:
                    raise ValueError(f"Duplicate ID in evaluation.csv: {row['eis_id']}")
                references[row["eis_id"]] = row
        logs = read_cpv_log(args.run / "extraction_terminal_logs.txt")
        outputs = sorted((args.run / "extracted").glob("*.json"))
        if not outputs:
            raise ValueError("No extraction JSON files found.")
        workers, cpv, model_seconds = [], [], 0
        for path in outputs:
            record = read_json(path)
            procurement_id = str(record["eis_id"])
            if procurement_id not in references or procurement_id not in logs:
                raise ValueError(f"Missing evaluation row or terminal-log section for {procurement_id}.")
            # The saved run name selects the matching trace, not another experiment.
            trace_name = Path(record["evaluation_criteria"]["trace_dir"]).name
            trace_dir = args.run / "criteria_worker_traces" / procurement_id / trace_name
            calls = read_json(trace_dir / "summary.json")["calls"]
            workers.extend(criteria_rows(record, read_json(trace_dir / "workers.json"), calls))
            cpv.append(cpv_row(record, references[procurement_id], logs[procurement_id]))
            model_seconds += sum(call["seconds"] for call in calls)
        if not workers:
            raise ValueError("No criteria workers found.")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    output = args.output or args.run / "orchestration_audit"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "criteria_workers.csv", workers)
    write_csv(output / "cpv_decisions.csv", cpv)
    report = summarize(workers, cpv, model_seconds)
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote summary.md, criteria_workers.csv and cpv_decisions.csv to {output}")


if __name__ == "__main__":
    main()
