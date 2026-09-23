"""Validate extracted fields against the open-data ground truth.

Read each extracted/{eis_id}.json (one file per procurement, written by the
pipeline) and match it to the same procurement inside data/raw_<date>.json (the
open-data download, which is a single JSON list of records). For every
procurement found on both sides, compare three fields and report how well the
extraction did:

  * main CPV code   - the single primary classification code
  * additional CPV  - the list of secondary codes
  * award criteria  - compared on their WEIGHTS, not their names (see below)

CPV and criteria are scored only for procurements whose ground truth
actually holds a value; where it does not, the procurement is skipped for that
field and listed in the summary, because there is nothing to judge agreement
against (these are the recovery cases, to read by hand).

Three outputs are produced:
  * a short precision / recall / F1 summary printed to the terminal
  * a JSON summary with coverage, denominators and CPV status strata
  * a CSV holding the raw extracted-vs-truth values for every procurement, so the
    summary numbers can be traced back to specific records by hand

Why criteria are scored on weights, not names: the same criterion is worded
differently across (and even within) documents, so the text is an unreliable
key. The weights (for example 30 / 30 / 40) are stable numbers, so they give a
language-independent comparison that ignores order. The CSV still prints both
sets of names side by side so a human can read them where the weights disagree.

Criteria retain decimal weights and lot labels. Their headline P/R/F1 is the
mean of per-procurement scores; CPV keeps micro scoring. Pooled criterion counts
and per-lot agreement are secondary diagnostics. Old unlabelled multi-lot
outputs are not assumed to apply to every lot.

Run:
    python validate_extraction.py
    python validate_extraction.py --ground-truth data/raw_2024-06-15.json
    python validate_extraction.py --csv my_report.csv
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from criteria_lots import lot_key, group_criteria, scope_error


# CONFIGURATION

EXTRACTED_DIR = Path("extracted")               # folder of per-procurement extraction files
DATA_DIR = Path("data")                         # folder holding the raw_<date>.json downloads


# GROUND TRUTH
# The open-data file is a JSON list of records, one per procurement. We need only
# three fields from each record, plus the procurement id to match the two sides on.

def simplify_ground_truth(record):
    """Return (eis_id, fields) for one raw record, or (None, _) if it has no URL.

    The procurement id is not a plain top-level field: it is the number at the end
    of tenderingProcess.documentsURL (for example .../Procurement/125861). The
    award criteria live one level down, inside each lot, as lots[].criterion[].
    """
    # The eis_id sits inside the documents URL, so pull it out with a regex.
    url = (record.get("tenderingProcess") or {}).get("documentsURL") or ""
    match = re.search(r"/Procurement/(\d+)", url)

    lots = []
    for index, lot in enumerate(record.get("lots") or []):
        label = lot.get("sequenceNumber")
        # Preserve anonymous lots as separate records; do not invent sequence numbers.
        lots.append({"lot": str(label) if label is not None else None,
                     "criteria": [{"name": c.get("winnerCriterionName") or "",
                                   "weight": c.get("winnerCriterionNumber"),
                                   "lot": str(label) if label is not None else None}
                                  for c in lot.get("criterion") or []]})
    fields = {"main_cpv": record.get("cpvType"),
              "additional_cpv": record.get("additionalCpvType") or [],
              "criteria": [c for lot in lots for c in lot["criteria"]],
              "criteria_lots": lots}

    return (match.group(1) if match else None), fields


def load_ground_truth(path, ids=None):
    """Load reference fields, optionally only for the requested extraction IDs."""
    selected = set(map(str, ids)) if ids is not None else None
    truth = {}
    # json.loads returns a Python list here, because the file is a JSON array.
    positions = {}
    for index, record in enumerate(json.loads(Path(path).read_text(encoding="utf-8")), 1):
        eis_id, fields = simplify_ground_truth(record)
        if eis_id and (selected is None or eis_id in selected):
            if eis_id in truth:
                raise ValueError(f"Duplicate reference ID {eis_id} at records {positions[eis_id]} and {index}; resolve duplicates before scoring.")
            positions[eis_id] = index
            truth[eis_id] = fields
    return truth


# EXTRACTED
# Each extracted/*.json holds one procurement in an already-flat shape. We pull
# the same three fields so both sides line up for a field-by-field comparison.

def load_extracted(folder):
    """Return {eis_id: fields} from the pipeline's extracted/*.json files."""
    out = {}
    for path in sorted(Path(folder).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        cpv = record.get("extracted") or {}             # the CPV block
        crit = record.get("evaluation_criteria") or {}  # the criteria block
        # Fall back to the file stem (125861 from 125861.json) if eis_id is absent.
        eis_id = str(record.get("eis_id") or path.stem)
        if eis_id in out:
            raise ValueError(f"Duplicate extraction ID {eis_id} in {path}; select one run before scoring.")
        out[eis_id] = {
            "status": cpv.get("status") or "legacy_unlabelled",
            "main_cpv": cpv.get("main_cpv"),
            "additional_cpv": cpv.get("additional_cpv") or [],
            "same_for_all_lots": crit.get("same_for_all_lots", False),
            "lot_ids": crit.get("lot_ids") or [],
            "lot_checks": crit.get("lot_checks") or {},
            "weight_scale": crit.get("weight_scale"),
            "trace_dir": crit.get("trace_dir", ""),
            "criteria": crit.get("criteria") or [],     # each item keeps name + weight
        }
    return out


# SCORING
# Every field is scored into three running counts:
#   tp = true positive  (predicted and correct)
#   fp = false positive (predicted but wrong, or predicted when nothing was there)
#   fn = false negative (was there but missed, or predicted wrongly)

def parsed_weight_values(raw):
    """Parse nonnegative weights without rounding, for reference eligibility.

    Decimal commas are supported. Numeric lists
    may use spaces, semicolons, pipes or slashes (legacy list syntax). Explicit
    '50 out of 100' means 50 points, without percentage conversion. Reject a
    whole field containing a negative/nonfinite value or unrecognised prose.
    """
    if raw is None or isinstance(raw, bool):
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)] if math.isfinite(raw) and raw >= 0 else []
    text = str(raw).strip()
    number = r"[+\-−]?\d+(?:[.,]\d+)?"
    ratio = re.fullmatch(rf"({number})\s+(?:points?\s+)?out\s+of\s+({number})", text, re.IGNORECASE)
    if ratio:
        numerator, denominator = [float(v.replace(",", ".").replace("−", "-")) for v in ratio.groups()]
        return [numerator] if (math.isfinite(numerator) and math.isfinite(denominator)
                                     and 0 <= numerator <= denominator and denominator > 0) else []
    # Validate the complete string so incidental numbers do not become weights.
    if not re.fullmatch(rf"{number}\s*%?(?:[\s;|/]+{number}\s*%?)*", text):
        return []
    values = [float(v.replace(",", ".").replace("−", "-")) for v in re.findall(number, text)]
    return values if all(math.isfinite(v) and v >= 0 for v in values) else []


def weight_values(raw):
    """Legacy integer helper retained for callers; lot scoring uses decimals."""
    return [round(value) for value in parsed_weight_values(raw)]


def weights(criteria):
    """Return the sorted criterion weights, dropping any criterion without one.

    Sorting makes the two lists comparable whatever order the criteria were listed
    in, so the comparison ignores order; weight_values parses each weight robustly.
    """
    return sorted(w for c in criteria for w in weight_values(c.get("weight")))


def score_scalar(pred, gold, counts):
    """Tally one single-valued field (the main CPV code) into counts.

    A wrong value counts as BOTH a false positive (we emitted a wrong code) and a
    false negative (we missed the right one). That is the usual way to score a
    single-label prediction, so that precision and recall both feel the error.
    """
    if gold and pred:
        if pred == gold:
            counts["tp"] += 1           # right code
        else:
            counts["fp"] += 1           # a wrong code was emitted
            counts["fn"] += 1           # the correct code was missed
    elif gold:                          # truth has a code, we produced nothing
        counts["fn"] += 1
    elif pred:                          # we produced a code, truth has none
        counts["fp"] += 1
    # if neither side has a value, there is nothing to score


def score_multiset(pred, gold, counts):
    """Tally a multiset field (additional CPV codes, or criterion weights).

    A multiset is a list that may hold duplicates (two criteria can both weigh
    30). Counter(pred) & Counter(gold) keeps the per-value minimum of the two,
    so its total is how many items the lists share - the true positives. Whatever
    is left unmatched on each side becomes a false positive or false negative.
    """
    tp = sum((Counter(pred) & Counter(gold)).values())
    counts["tp"] += tp
    counts["fp"] += len(pred) - tp      # extra items we predicted but truth lacked
    counts["fn"] += len(gold) - tp      # items in truth we did not predict


def prf(counts):
    """Return (precision, recall, f1) from a tp/fp/fn tally.

    precision = of what we predicted, how much was right
    recall    = of what was actually there, how much we found
    f1        = the harmonic mean, which stays low unless both are high
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# REPORTING

def write_csv(path, rows):
    """Write one detailed comparison row per procurement to a CSV file."""
    columns = ["eis_id", "output_present", "reference_present", "cpv_status",
               "criteria_reference", "criteria_reference_total", "criteria_expected_total", "criteria_extraction_complete",
               "main_cpv_extracted", "main_cpv_truth", "main_cpv_match",
               "additional_cpv_extracted", "additional_cpv_truth", "additional_cpv_match",
               "weights_extracted", "weights_truth", "weights_match",
               "criteria_names_extracted", "criteria_names_truth", "criteria_lot_details"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# EVALUATION

FIELD_NAMES = ("main_cpv", "additional_cpv", "criteria")
EMPTY = {"main_cpv": None, "additional_cpv": [], "criteria": []}


def new_metrics():
    return {name: {"tp": 0, "fp": 0, "fn": 0, "records": 0, "exact_matches": 0}
            for name in FIELD_NAMES}


def complete_weights(criteria):
    """Check weight presence and parsing, not reference eligibility or correctness."""
    return bool(criteria) and all(weight_values(c.get("weight")) for c in criteria)


def criteria_reference(criteria, expected_total=100.0, tolerance=0.5):
    """Apply the declared total rule to one reference lot before scoring.

    This determines eligibility for this evaluation, not whether a document is
    correct. A legitimate different scoring scale needs a different cohort rule.
    Predictions never determine reference eligibility.
    """
    if not criteria:
        return {"status": "missing", "total": None}
    groups = [parsed_weight_values(c.get("weight")) for c in criteria]
    if not all(groups):
        return {"status": "incomplete", "total": None}
    total = math.fsum(v for group in groups for v in group)
    return {"status": "available" if abs(total - expected_total) <= tolerance else "total_mismatch",
            "total": total}


def reference_lots(record):
    """Keep empty reference lots, which cannot be eligible by omission."""
    if "criteria_lots" in record:
        return record["criteria_lots"]
    return [{"lot": None if key == "all" else key, "criteria": items}
            for key, items in group_criteria(record.get("criteria", [])).items()]


def lot_reference(record, expected_total=100.0, tolerance=0.5):
    lots = reference_lots(record)
    checks = [criteria_reference(lot["criteria"], expected_total, tolerance) for lot in lots]
    status = next((c["status"] for c in checks if c["status"] != "available"),
                  "available" if checks else "missing")
    totals = [c["total"] for c in checks]
    return {"status": status, "total": math.fsum(totals) if totals and None not in totals else None,
            "lots": [{"lot": lot.get("lot"), **check} for lot, check in zip(lots, checks)]}


def decimal_weights(items):
    """Keep parsed weights without integer or decimal-place rounding."""
    return sorted(w for c in items for w in parsed_weight_values(c.get("weight")))



WEIGHT_MATCH_TOLERANCE = 0.01  # Absolute points, separate from the 0.5 total check.


def score_weights(pred, gold, counts):
    """Match sorted weights one-to-one within 0.01 points, including duplicates."""
    predicted = sorted(Decimal(str(w)) for w in pred)
    reference = sorted(Decimal(str(w)) for w in gold)
    tolerance = Decimal(str(WEIGHT_MATCH_TOLERANCE))
    i = j = matches = 0
    while i < len(predicted) and j < len(reference):
        if abs(predicted[i] - reference[j]) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif predicted[i] < reference[j]:
            i += 1
        else:
            j += 1
    counts["tp"] += matches
    counts["fp"] += len(predicted) - matches
    counts["fn"] += len(reference) - matches
    return matches

def scoring_copy(extracted, stated_only=False):
    """Keep the stored record intact; convert only explicitly fractional scales."""
    items = []
    for original in extracted.get("criteria") or []:
        c = dict(original)
        value = c.get("raw_weight", c.get("weight")) if stated_only else c.get("weight")
        scale = (extracted.get("lot_checks", {}).get(lot_key(c.get("lot"))) or {}).get(
            "weight_scale", extracted.get("weight_scale"))
        if scale == "fraction":
            values = parsed_weight_values(value)
            c["weight"] = " ".join(str(v * 100) for v in values) if values else None
        else:
            c["weight"] = value
        items.append(c)
    return {**extracted, "criteria": items, "weight_scale": "points", "lot_checks": {}}


def write_names_review(path, truth, extracted, ids):
    """Export names independently of weight eligibility; blanks require manual review."""
    columns = ["eis_id", "lot", "reference_names", "extracted_names", "duplicate_names",
               "weight_sources", "trace_dir", "names_complete", "lot_scope_supported", "notes"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for eid in ids:
            e = extracted[eid]
            pred = group_criteria(e.get("criteria") or [])
            gold = {lot_key(l["lot"]): l["criteria"] for l in reference_lots(truth.get(eid, {}))}
            for label in sorted(set(pred) | set(gold)):
                names = [c.get("name", "") for c in pred.get(label, [])]
                normalized = [" ".join(n.casefold().split()) for n in names]
                writer.writerow({"eis_id": eid, "lot": label,
                    "reference_names": json.dumps([c.get("name") for c in gold.get(label, [])], ensure_ascii=False),
                    "extracted_names": json.dumps(names, ensure_ascii=False),
                    "duplicate_names": len(normalized) != len(set(normalized)),
                    "weight_sources": json.dumps([c.get("weight_source", "legacy_unspecified") for c in pred.get(label, [])]),
                    "trace_dir": e.get("trace_dir", ""),
                    "names_complete": "", "lot_scope_supported": "", "notes": ""})


def compare_criteria(extracted, truth):
    """Compare by lot label; use unlabeled signatures only as a secondary measure.

    A single unlabelled set is NOT expanded unless the extraction explicitly
    says it applies to every lot. Reference values never supply missing weights.
    """
    extracted = scoring_copy(extracted)
    gold = reference_lots(truth)
    items = extracted.get("criteria") or []
    shared = extracted.get("same_for_all_lots", False)
    problem = scope_error(items, shared, extracted.get("lot_ids") or [])
    grouped = group_criteria(items)
    pred = [{"lot": None if key == "all" else key, "criteria": value} for key, value in grouped.items()]
    declared = {lot_key(x) for x in extracted.get("lot_ids", [])} - {"all"}
    gold_labels = [lot_key(lot.get("lot")) for lot in gold]
    known_labels = set(gold_labels) - {"all"}
    inventory_assessable = len(known_labels) == len(gold_labels) and bool(gold_labels)
    unknown_lots = sorted(declared - known_labels) if inventory_assessable else []
    if unknown_lots:
        problem = "The extracted inventory contains unknown lots: " + ", ".join(unknown_lots)
    inventory_complete = declared == known_labels if inventory_assessable else None
    if shared and items and not problem:
        pred = [{"lot": lot.get("lot"), "criteria": items} for lot in gold]
    gold_signatures = [tuple(decimal_weights(lot["criteria"])) for lot in gold]
    pred_signatures = [tuple(decimal_weights(lot["criteria"])) for lot in pred
                       if complete_weights(lot["criteria"])]
    unordered_matches = sum((Counter(gold_signatures) & Counter(pred_signatures)).values())
    counts = {"tp": 0, "fp": 0, "fn": 0}
    pred_labels = [lot_key(lot.get("lot")) for lot in pred]
    labelled = (all(k != "all" for k in gold_labels + pred_labels)
                and len(set(gold_labels)) == len(gold_labels)
                and len(set(pred_labels)) == len(pred_labels))
    pairs = []
    mode = "by_lot_label"
    if problem:
        mode = "invalid_scope"
    elif len(gold) == len(pred) == 1:
        if gold_labels[0] == pred_labels[0] or "all" in (gold_labels[0], pred_labels[0]):
            pairs = [(gold[0], pred[0])]
            mode = "single_set"
        else:
            pairs = [(gold[0], None), (None, pred[0])]
    elif labelled:
        gm = dict(zip(gold_labels, gold)); pm = dict(zip(pred_labels, pred))
        pairs = [(gm.get(key), pm.get(key)) for key in sorted(set(gm) | set(pm))]
    else:
        mode = "unresolved_lot_alignment"
    if not pairs:
        pairs = [(lot, None) for lot in gold] + [(None, lot) for lot in pred]
    matched_lots = 0
    for g, e in pairs:
        gw = decimal_weights(g["criteria"]) if g else []
        ew = decimal_weights(e["criteria"]) if e else []
        matches = score_weights(ew, gw, counts)
        matched_lots += int(bool(g and e) and matches == len(gw) == len(ew) and complete_weights(e["criteria"]))
    exact = bool(gold) and matched_lots == len(gold) == len(pred) and not problem
    return {**counts, "exact": exact, "mode": mode, "scope_error": problem,
            "inventory_complete": inventory_complete, "declared_lots": sorted(declared),
            "unlisted_reference_lots": sorted(known_labels - declared) if inventory_assessable else None,
            "unknown_declared_lots": unknown_lots,
            "weight_match_tolerance": WEIGHT_MATCH_TOLERANCE,
            "unordered_weight_policy": "strict decimal equality; diagnostic only",
            "reference_lots": len(gold), "extracted_lots": len(pred),
            "matched_lots": matched_lots, "unordered_matching_lots": unordered_matches,
            "unordered_exact": bool(gold) and unordered_matches == len(gold) == len(pred) and not problem,
            "reference": [{"lot": lot.get("lot"), "weights": decimal_weights(lot["criteria"])} for lot in gold],
            "extracted": [{"lot": lot.get("lot"), "weights": decimal_weights(lot["criteria"])} for lot in pred]}


def add_scores(metrics, e, g, expected_total=100.0, tolerance=0.5):
    """CPV remains micro-scored; criteria uses equal-procurement macro averages."""
    pairs = {"main_cpv": ((e.get("main_cpv") or "").strip(), (g.get("main_cpv") or "").strip()),
             "additional_cpv": ([c.strip() for c in e.get("additional_cpv", [])],
                                [c.strip() for c in g.get("additional_cpv", [])])}
    for name, (pred, gold) in pairs.items():
        if not gold:
            continue
        tally = metrics[name]
        tally["records"] += 1
        tally["exact_matches"] += int(pred == gold if name == "main_cpv" else sorted(pred) == sorted(gold))
        (score_scalar if name == "main_cpv" else score_multiset)(pred, gold, tally)
    if lot_reference(g, expected_total, tolerance)["status"] != "available":
        return
    result = compare_criteria(e, g)
    tally = metrics["criteria"]
    tally["records"] += 1
    tally["exact_matches"] += int(result["exact"])
    for key in ("tp", "fp", "fn"):
        tally[key] += result[key]
    for key, value in zip(("precision_sum", "recall_sum", "f1_sum"), prf(result)):
        tally[key] = tally.get(key, 0.0) + value
    for key in ("reference_lots", "extracted_lots", "matched_lots", "unordered_matching_lots"):
        tally[key] = tally.get(key, 0) + result[key]


def finish_metrics(metrics):
    for name, counts in metrics.items():
        p, r, f = prf(counts)
        if name == "criteria":
            counts["micro_diagnostic"] = dict(zip(("precision", "recall", "f1"), (p, r, f)))
            n = counts["records"]
            p, r, f = [counts.get(k, 0) / n if n else 0 for k in ("precision_sum", "recall_sum", "f1_sum")]
            counts["averaging"] = "macro_per_procurement"
            total_lots = counts.get("reference_lots", 0)
            counts["lot_exact_match_rate"] = counts.get("matched_lots", 0) / total_lots if total_lots else None
        else:
            counts["averaging"] = "micro"
        scoreable = bool(counts["records"])
        counts.update(precision=p if scoreable else None, recall=r if scoreable else None,
                      f1=f if scoreable else None,
                      exact_match_rate=counts["exact_matches"] / counts["records"] if scoreable else None)
    return metrics


def evaluate(truth, extracted, ids=None, expected_total=100.0, tolerance=0.5):
    """Evaluate output IDs only, optionally narrowed by an explicit ID list.

    An existing output with empty fields is still scored against eligible
    references. IDs without an output never enter the scoring denominator.
    """
    if not math.isfinite(expected_total) or expected_total <= 0 or not math.isfinite(tolerance) or not 0 <= tolerance < expected_total:
        raise ValueError("Criteria total must be positive and finite; tolerance must be finite and between zero and the total (exclusive).")
    requested = set(map(str, ids)) if ids is not None else set(extracted)
    selected = requested & set(extracted)
    shared = selected & set(truth) & set(extracted)
    reference_ids = selected & set(truth)
    matched = new_metrics()
    stated = new_metrics()
    strata, rows = {}, []
    for eis_id in sorted(selected):
        e, g = extracted.get(eis_id, EMPTY), truth.get(eis_id, EMPTY)
        present, referenced = eis_id in extracted, eis_id in truth
        status = (e.get("status") or "legacy_unlabelled") if present else "missing_output"
        if present:
            stratum = strata.setdefault(status, {"outputs": 0, "matched_references": 0, "metrics": new_metrics()})
            stratum["outputs"] += 1
            if referenced:
                stratum["matched_references"] += 1
                # CPV-only strata: don't condition criteria scores on another branch.
                add_scores(stratum["metrics"], {**e, "criteria": []}, {**g, "criteria": [], "criteria_lots": []})
        if eis_id in shared:
            add_scores(matched, e, g, expected_total, tolerance)
            add_scores(stated, scoring_copy(e, stated_only=True), g, expected_total, tolerance)
        gc, ec = g.get("criteria", []), e.get("criteria", [])
        g_w, e_w = decimal_weights(gc), decimal_weights(scoring_copy(e)["criteria"])
        reference = lot_reference(g, expected_total, tolerance)
        comparison = compare_criteria(e, g)
        criterion_reference = reference["status"]
        gm, em = (g.get("main_cpv") or "").strip(), (e.get("main_cpv") or "").strip()
        ga, ea = [c.strip() for c in g.get("additional_cpv", [])], [c.strip() for c in e.get("additional_cpv", [])]
        rows.append({"eis_id": eis_id, "output_present": present, "reference_present": referenced,
                     "cpv_status": status, "criteria_reference": criterion_reference,
                     "criteria_reference_total": reference["total"], "criteria_expected_total": expected_total,
                     "criteria_extraction_complete": complete_weights(ec),
                     "main_cpv_extracted": em, "main_cpv_truth": gm,
                     "main_cpv_match": em == gm if gm else "",
                     "additional_cpv_extracted": "; ".join(ea), "additional_cpv_truth": "; ".join(ga),
                     "additional_cpv_match": sorted(ea) == sorted(ga) if ga else "",
                     "weights_extracted": "; ".join(map(str, e_w)), "weights_truth": "; ".join(map(str, g_w)),
                     "weights_match": comparison["exact"] if criterion_reference == "available" else "",
                     "criteria_names_extracted": " | ".join(c.get("name") or "" for c in ec),
                     "criteria_names_truth": " | ".join(c.get("name") or "" for c in gc),
                     "criteria_lot_details": json.dumps({"reference_checks": reference["lots"], **comparison}, ensure_ascii=False)})
    for stratum in strata.values():
        stratum["metrics"] = {k: v for k, v in finish_metrics(stratum["metrics"]).items() if k != "criteria"}
    summary = {"criteria_reference_policy": {"expected_total": expected_total, "tolerance": tolerance, "round_before_check": False, "unit": "each lot; all lots must be eligible"},
               "weight_policy": "criteria one-to-one matching within 0.01 absolute points; per-procurement macro P/R/F1; label-aware primary comparison",
               "scope": "extracted_ids_filtered" if ids is not None else "extracted_ids",
               "requested_ids_without_output": sorted(requested - set(extracted)),
               "selected_ids": sorted(selected),
               "coverage": {"selected": len(selected), "references": len(reference_ids),
                            "outputs": len(selected & set(extracted)), "matched": len(shared),
                            "missing_reference_ids": sorted((selected & set(extracted)) - set(truth)),
                            },
               "reference_coverage": {name: {"available": matched[name]["records"],
                                             "unavailable": len(reference_ids) - matched[name]["records"]}
                                      for name in FIELD_NAMES},
               "matched_records": finish_metrics(matched),
               "cpv_status_strata": strata,
               "criteria_stated_only": finish_metrics(stated)["criteria"],
               "defaults_applied": sum(c.get("weight_source") == "sole_criterion_default"
                   for eid in selected for c in extracted[eid].get("criteria", [])),
               "name_policy": "Separate manual semantic review; weight agreement does not establish name accuracy."}
    summary["worklists"] = {
        "criteria_mismatch_ids": [r["eis_id"] for r in rows if r["output_present"] and r["weights_match"] is False],
        "criteria_reference_excluded": {
            reason: [r["eis_id"] for r in rows if r["reference_present"] and r["criteria_reference"] == reason]
            for reason in ("missing", "incomplete", "total_mismatch")}}
    return rows, summary


def audit_ground_truth(path):
    """List every repeated procurement ID before evaluation, without deduping."""
    groups = {}
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    missing_id = []
    for position, record in enumerate(records, 1):
        eis_id, fields = simplify_ground_truth(record)
        if eis_id is None:
            missing_id.append(position)
        else:
            groups.setdefault(eis_id, []).append((position, fields))
    duplicates = [{"eis_id": eid, "record_positions": [p for p, _ in entries],
                   "compared_fields_identical": all(f == entries[0][1] for _, f in entries[1:])}
                  for eid, entries in sorted(groups.items()) if len(entries) > 1]
    return {"records": len(records), "unique_ids": len(groups), "duplicate_ids": duplicates,
            "records_without_id": missing_id}


def print_metrics(label, metrics):
    """Print declared averaging and per-procurement exact match."""
    print(f"\n{label}")
    print("CPV: micro P/R/F1. Criteria: macro P/R/F1 per procurement; pooled tp/fp/fn are diagnostics only.")
    print(f"{'Field':<18} {'Records':>7} {'Exact match':>11} {'P':>6} {'R':>6} {'F1':>6}  tp/fp/fn")
    for field, c in metrics.items():
        values = [f"{c[k]:.2f}" if c[k] is not None else "-" for k in ("exact_match_rate", "precision", "recall", "f1")]
        print(f"{field:<18} {c['records']:>7} {values[0]:>11} {values[1]:>6} {values[2]:>6} {values[3]:>6}  {c['tp']}/{c['fp']}/{c['fn']}")


def main():
    """Write CSV details and a JSON summary with explicit scoring denominators."""
    parser = argparse.ArgumentParser(description="Validate extracted fields against open data.")
    parser.add_argument("--extracted", default=str(EXTRACTED_DIR))
    parser.add_argument("--ground-truth", help="default: newest data/raw_*.json")
    parser.add_argument("--csv", help="default: validation_<folder>.csv")
    parser.add_argument("--summary", help="default: CSV filename with .summary.json suffix")
    parser.add_argument("--ids-file", help="One procurement ID per line")
    parser.add_argument("--ids", nargs="+", help="restrict evaluation to these IDs with extraction outputs")
    parser.add_argument("--criteria-total", type=float, default=100.0, help="reference weight total required for this cohort (default 100)")
    parser.add_argument("--criteria-tolerance", type=float, default=0.5, help="absolute tolerance on unrounded reference weights (default 0.5)")
    parser.add_argument("--check-reference", action="store_true", help="audit duplicate reference IDs only, without loading outputs or scoring")
    args = parser.parse_args()
    if args.ids_file:
        args.ids = list(args.ids or []) + Path(args.ids_file).read_text(encoding="utf-8").split()
    csv_path = Path(args.csv or f"validation_{Path(args.extracted).name}.csv")
    summary_path = Path(args.summary) if args.summary else csv_path.with_suffix(".summary.json")
    if csv_path.resolve() == summary_path.resolve():
        parser.error("--csv and --summary must be different paths")
    gt_path = args.ground_truth
    if not gt_path:
        found = sorted(DATA_DIR.glob("raw_*.json"), reverse=True)
        if not found:
            parser.error(f"No raw_*.json in {DATA_DIR}. Pass --ground-truth.")
        gt_path = found[0]
    if args.check_reference:
        audit = audit_ground_truth(gt_path)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        if audit["duplicate_ids"]:
            parser.exit(2, "Duplicate reference IDs found; resolve upstream before evaluation. No records were changed.\n")
        return audit
    try:
        extracted = load_extracted(args.extracted)
        selected = set(extracted) if args.ids is None else set(extracted) & set(args.ids)
        truth = load_ground_truth(gt_path, selected)
    except ValueError as error:
        parser.error(str(error))
    try:
        rows, summary = evaluate(truth, extracted, args.ids, args.criteria_total, args.criteria_tolerance)
    except ValueError as error:
        parser.error(str(error))
    summary.update(ground_truth=str(gt_path), extracted_folder=str(args.extracted))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    review_path = csv_path.with_suffix(".names_review.csv")
    if review_path.resolve() in {csv_path.resolve(), summary_path.resolve()}:
        parser.error("Review CSV must differ from output paths")
    write_names_review(review_path, truth, extracted, summary["selected_ids"])
    write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    c = summary["coverage"]
    print(f"Evaluating {c['outputs']} extraction outputs; matched references: {c['matched']}; "
          f"outputs without reference: {len(c['missing_reference_ids'])}.")
    if summary["requested_ids_without_output"]:
        print("Requested IDs without extraction outputs (excluded): " +
              ", ".join(summary["requested_ids_without_output"]))
    print_metrics("Agreement with eligible reference values", summary["matched_records"])
    print_metrics("Criteria using stated weights only", {"criteria": summary["criteria_stated_only"]})
    print("Defaulted sole-criterion weights:", summary["defaults_applied"])
    print("Names require manual review:", review_path)
    work = summary["worklists"]
    print("\nCriteria weights differ, check names by hand: " + (", ".join(work["criteria_mismatch_ids"]) or "none"))
    for reason, ids in work["criteria_reference_excluded"].items():
        print(f"Criteria references excluded ({reason}, {len(ids)}): " + (", ".join(ids) or "none"))
    print("\nCPV status strata (counts and scores in JSON):", {k: v["outputs"] for k, v in summary["cpv_status_strata"].items()})
    print("Reference coverage:", summary["reference_coverage"])
    print(f"Wrote {csv_path} and {summary_path}")
    return summary


if __name__ == "__main__":
    main()
