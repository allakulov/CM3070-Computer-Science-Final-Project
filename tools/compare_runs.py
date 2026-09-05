"""Turn the review run log into a table, one column per model.

Each run of review_standards.py appends a line to standards_review_runs.jsonl, so
this reads them back and lays the verdicts side by side: one row per candidate
standard, one column per model, plus the run totals underneath. Where the models
disagree the row is marked, because those are the ones worth reading by hand.

Run:
    python compare_runs.py
    python compare_runs.py --csv comparison.csv
"""

import argparse
import csv
import json
from pathlib import Path


# CONFIGURATION

RUNS_PATH = Path("standards_review_runs.jsonl")
MARK = {True: "yes", False: "no"}       # how a verdict is shown


def load_runs(path):
    """Read the run log, keeping the latest run per model.

    Returns:
        dict: model name mapped to its run.
    """
    runs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            run = json.loads(line)
            runs[run["model"]] = run
    return runs


def build_rows(runs):
    """Return the table rows: one per candidate, one column per model."""
    names = []
    for run in runs.values():
        for name in run["verdicts"]:
            if name not in names:
                names.append(name)

    rows = []
    for name in sorted(names):
        row = {"standard": name}
        verdicts = []
        for model, run in runs.items():
            verdict = run["verdicts"].get(name)
            row[model] = MARK.get(verdict["applies"], "-") if verdict else "-"
            if verdict:
                verdicts.append(verdict["applies"])
        row["agree"] = "" if len(set(verdicts)) <= 1 else "differs"
        rows.append(row)
    return rows


TOTALS = [("candidates", "candidates"), ("applies", "applies"),
          ("paused", "pauses"), ("corrected", "overridden"),
          ("seconds each", "seconds_each"), ("seconds total", "seconds")]


def total_rows(models, runs):
    """Return the run totals shaped like the verdict rows, so both share a table."""
    return [{"standard": label, "agree": "",
             **{model: runs[model].get(key, "-") for model in models}}
            for label, key in TOTALS]


def print_table(rows, models, runs):
    """Print the table, then the run totals for each model."""
    width = max(len(row["standard"]) for row in rows) + 2
    header = f"{'standard':<{width}}" + "".join(f"{m:<16}" for m in models) + "agree"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['standard']:<{width}}" + "".join(f"{row[m]:<16}" for m in models)
        print(line + row["agree"])

    print()
    for row in total_rows(models, runs):
        print(f"{row['standard']:<{width}}" + "".join(f"{row[m]!s:<16}" for m in models))


def main():
    """Lay the review runs out as a table for comparison."""
    parser = argparse.ArgumentParser(description="Compare review runs model by model.")
    parser.add_argument("--runs", default=str(RUNS_PATH), help="the run log to read")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args()

    path = Path(args.runs)
    if not path.is_file():
        raise SystemExit(f"no run log at {path}; run review_standards.py first")

    runs = load_runs(path)
    models = list(runs)
    rows = build_rows(runs)
    print_table(rows, models, runs)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["standard"] + models + ["agree"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({})                       # blank line between the two halves
            writer.writerows(total_rows(models, runs))
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()