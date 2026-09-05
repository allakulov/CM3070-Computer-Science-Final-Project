"""Diff two models' extractions to find where they disagree.

Reads the extracted/{model}/{id}.json records for two runs and prints, per
procurement, the structural fields whose values differ, with each model's CPV
reasoning shown for context. Most procurements agree, so this points the hand-check
at the ones that do not.

Run:
    python compare_extractions.py extracted/mistral-small extracted/qwen3.5-9b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STRUCTURAL = ("main_cpv", "additional_cpv", "criteria_weights")


def _load(folder):
    """Read every {id}.json in a folder into {id: record}."""
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in Path(folder).glob("*.json")}


def _fields(record):
    """Pull the comparable fields out of one extracted record."""
    extracted = record.get("extracted") or {}
    criteria = record.get("evaluation_criteria") or {}
    weights = sorted(c["weight"] for c in (criteria.get("criteria") or []) if c.get("weight") is not None)
    return {"main_cpv": extracted.get("main_cpv"),
            "additional_cpv": sorted(extracted.get("additional_cpv") or []),
            "criteria_weights": weights,
            "reasoning": extracted.get("reasoning")}


def compare(folder_a, folder_b):
    """Print the procurements where the two runs disagree on a structural field."""
    a, b = _load(folder_a), _load(folder_b)
    name_a, name_b = Path(folder_a).name, Path(folder_b).name
    shared = sorted(set(a) & set(b))
    if not shared:
        print("  no procurements in common")
        return

    disagreed = 0
    for eis_id in shared:
        fa, fb = _fields(a[eis_id]), _fields(b[eis_id])
        diffs = [key for key in STRUCTURAL if fa[key] != fb[key]]
        if not diffs:
            continue
        disagreed += 1
        print(f"\n{eis_id}")
        for key in diffs:
            print(f"    {key}:  {name_a}={fa[key]}  |  {name_b}={fb[key]}")
        print(f"    reasoning [{name_a}]: {fa['reasoning']}")
        print(f"    reasoning [{name_b}]: {fb['reasoning']}")

    print(f"\n  {disagreed} of {len(shared)} procurements disagree structurally")
    secs_a = sum(a[i].get("model_seconds") or 0 for i in shared)
    secs_b = sum(b[i].get("model_seconds") or 0 for i in shared)
    n = len(shared)
    print(f"  model time over {n}: {name_a} {secs_a:.1f}s ({secs_a / n:.1f}s avg)"
          f"  |  {name_b} {secs_b:.1f}s ({secs_b / n:.1f}s avg)")


def main():
    parser = argparse.ArgumentParser(description="Diff two models' extractions field by field.")
    parser.add_argument("folder_a", help="e.g. extracted/mistral-small")
    parser.add_argument("folder_b", help="e.g. extracted/qwen3.5-9b")
    args = parser.parse_args()
    compare(args.folder_a, args.folder_b)


if __name__ == "__main__":
    main()