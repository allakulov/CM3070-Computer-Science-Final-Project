"""Validate extracted fields against the open-data ground truth.

Read each extracted/{eis_id}.json (one file per procurement, written by the
pipeline) and match it to the same procurement inside data/raw_<date>.json (the
open-data download, which is a single JSON list of records). For every
procurement found on both sides, compare three fields and report how well the
extraction did:

  * main CPV code   - the single primary classification code
  * additional CPV  - the list of secondary codes
  * award criteria  - compared on their WEIGHTS, not their names (see below)

Main and additional CPV are scored only for procurements whose ground truth
actually holds a value; where it does not, the procurement is skipped for that
field and listed in the summary, because there is nothing to judge agreement
against (these are the recovery cases, to read by hand).

Two outputs are produced:
  * a short precision / recall / F1 summary printed to the terminal
  * a CSV holding the raw extracted-vs-truth values for every procurement, so the
    summary numbers can be traced back to specific records by hand

Why criteria are scored on weights, not names: the same criterion is worded
differently across (and even within) documents, so the text is an unreliable
key. The weights (for example 30 / 30 / 40) are stable numbers, so they give a
language-independent comparison that ignores order. The CSV still prints both
sets of names side by side so a human can read them where the weights disagree.

Run:
    python validate_extraction.py
    python validate_extraction.py --ground-truth data/raw_2024-06-15.json
    python validate_extraction.py --csv my_report.csv
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


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

    # Collect every criterion across every lot. Most procurements have a single
    # lot, but looping over all lots keeps this correct when there are several.
    criteria = [
        {
            "name": c.get("winnerCriterionName") or "",
            "weight": c.get("winnerCriterionNumber"),   # this number is the weight
        }
        for lot in record.get("lots") or []
        for c in lot.get("criterion") or []
    ]

    fields = {
        "main_cpv": record.get("cpvType"),                         # primary code
        "additional_cpv": record.get("additionalCpvType") or [],   # secondary codes
        "criteria": criteria,
    }
    return (match.group(1) if match else None), fields


def load_ground_truth(path):
    """Return {eis_id: fields} built from the open-data list file."""
    truth = {}
    # json.loads returns a Python list here, because the file is a JSON array.
    for record in json.loads(Path(path).read_text(encoding="utf-8")):
        eis_id, fields = simplify_ground_truth(record)
        if eis_id:                      # skip any record we could not assign an id to
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
        out[record.get("eis_id") or path.stem] = {
            "main_cpv": cpv.get("main_cpv"),
            "additional_cpv": cpv.get("additional_cpv") or [],
            "criteria": crit.get("criteria") or [],     # each item keeps name + weight
        }
    return out


# SCORING
# Every field is scored into three running counts:
#   tp = true positive  (predicted and correct)
#   fp = false positive (predicted but wrong, or predicted when nothing was there)
#   fn = false negative (was there but missed, or predicted wrongly)

def weight_values(raw):
    """Return the weight(s) in one criterion field as rounded ints, robust to format.

    A field is normally a number, but the ground truth sometimes stores it as a
    string that formats or separates values differently. Every number is pulled
    out, so a list separator (; | / or a space) between weights is handled, and a
    comma sitting inside a number is read as a decimal point (30,5 becomes 30.5).
    """
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [round(raw)]
    return [round(float(number.replace(",", ".")))
            for number in re.findall(r"\d+(?:[.,]\d+)?", str(raw))]


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
    columns = [
        "eis_id",
        "main_cpv_extracted", "main_cpv_truth", "main_cpv_match",
        "additional_cpv_extracted", "additional_cpv_truth", "additional_cpv_match",
        "weights_extracted", "weights_truth", "weights_match",
        "criteria_names_extracted", "criteria_names_truth",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# MAIN

def main():
    """Compare extracted files to the ground truth, print a summary, write a CSV."""
    parser = argparse.ArgumentParser(description="Validate extracted fields against open data.")
    parser.add_argument("--extracted", default=str(EXTRACTED_DIR),
                        help="folder of extracted/*.json files")
    parser.add_argument("--ground-truth",
                        help="open-data file; default: newest data/raw_*.json")
    parser.add_argument("--csv", default=None,
                        help="detailed table output; default: validation_<extracted-folder>.csv")
    args = parser.parse_args()

    # Name the CSV after the extracted folder, so each model's run keeps its own file.
    csv_path = args.csv or f"validation_{Path(args.extracted).name}.csv"

    # Pick the ground-truth file: the one passed in, or the newest raw_*.json.
    gt_path = args.ground_truth
    if not gt_path:
        found = sorted(DATA_DIR.glob("raw_*.json"), reverse=True)
        if not found:
            print(f"No raw_*.json in {DATA_DIR}. Pass --ground-truth.")
            return
        gt_path = found[0]

    truth = load_ground_truth(gt_path)
    extracted = load_extracted(args.extracted)

    # Only procurements present on BOTH sides can be compared field by field.
    shared = sorted(set(truth) & set(extracted))

    # One running tally per field type.
    fields = {name: {"tp": 0, "fp": 0, "fn": 0}
              for name in ("main_cpv", "additional_cpv", "criteria")}
    skipped_main = []       # ids with no ground-truth main CPV, skipped for that field
    skipped_additional = [] # ids with no ground-truth additional CPV, skipped for that field
    review = []             # ids whose criterion weights disagree, for a human to read
    rows = []               # one detailed CSV row per procurement

    for eis_id in shared:
        e, g = extracted[eis_id], truth[eis_id]

        # MAIN CPV: score only where the ground truth holds a code; skip otherwise.
        e_main = (e["main_cpv"] or "").strip()
        g_main = (g["main_cpv"] or "").strip()
        if g_main:
            score_scalar(e_main, g_main, fields["main_cpv"])
        else:
            skipped_main.append(eis_id)

        # ADDITIONAL CPV: score only where the ground truth holds codes; skip otherwise.
        e_add = [c.strip() for c in e["additional_cpv"]]
        g_add = [c.strip() for c in g["additional_cpv"]]
        if g_add:
            score_multiset(e_add, g_add, fields["additional_cpv"])
        else:
            skipped_additional.append(eis_id)

        # CRITERIA: compare on weights only, as an order-independent multiset;
        # names are too variable to score on and are read by hand from the CSV.
        e_w, g_w = weights(e["criteria"]), weights(g["criteria"])
        score_multiset(e_w, g_w, fields["criteria"])
        if e_w != g_w:
            review.append(eis_id)

        # Gather the raw values for this procurement into one CSV row. Lists are
        # joined into single strings so each value occupies a single cell.
        rows.append({
            "eis_id": eis_id,
            "main_cpv_extracted": e_main,
            "main_cpv_truth": g_main,
            "main_cpv_match": e_main == g_main,
            "additional_cpv_extracted": "; ".join(e_add),
            "additional_cpv_truth": "; ".join(g_add),
            "additional_cpv_match": sorted(e_add) == sorted(g_add),
            "weights_extracted": "; ".join(str(w) for w in e_w),
            "weights_truth": "; ".join(str(w) for w in g_w),
            "weights_match": e_w == g_w,
            "criteria_names_extracted": " | ".join(c.get("name") or "" for c in e["criteria"]),
            "criteria_names_truth": " | ".join(c.get("name") or "" for c in g["criteria"]),
        })

    # SUMMARY (terminal)
    print(f"Ground truth: {gt_path}")
    print(f"Evaluated {len(shared)} procurements "
          f"(unmatched: {len(set(extracted) - set(truth))} extracted, "
          f"{len(set(truth) - set(extracted))} ground truth)\n")
    print(f"{'field':<16} {'P':>5} {'R':>5} {'F1':>5}   tp/fp/fn")
    for name, c in fields.items():
        if c["tp"] + c["fp"] + c["fn"] == 0:        # nothing of this field anywhere
            print(f"{name:<16} {'-':>5} {'-':>5} {'-':>5}   (none)")
        else:
            p, r, f = prf(c)
            print(f"{name:<16} {p:>5.2f} {r:>5.2f} {f:>5.2f}   {c['tp']}/{c['fp']}/{c['fn']}")

    if skipped_main:
        print(f"\nSkipped main_cpv, no ground-truth code ({len(skipped_main)}): "
              f"{', '.join(skipped_main)}")
    if skipped_additional:
        print(f"Skipped additional_cpv, no ground-truth codes ({len(skipped_additional)}): "
              f"{', '.join(skipped_additional)}")
    if review:
        print(f"\nCriteria weights differ, check names by hand: {', '.join(review)}")

    # DETAILED TABLE (csv)
    if rows:
        write_csv(csv_path, rows)
        print(f"\nWrote detailed comparison to {csv_path}")


if __name__ == "__main__":
    main()