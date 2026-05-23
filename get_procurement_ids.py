"""Download procurement data from open.iub.gov.lv and list available fields.

Usage::

    python get_procurement_ids.py
    python get_procurement_ids.py --date 2024-06-15
"""

import json
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


def download_daily_json(date_str):
    """Download the daily procurement JSON for a given date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    url = (
        f"https://open.iub.gov.lv/data/notice/"
        f"{dt.year}/{dt.month:02d}/{dt.day:02d}-{dt.month:02d}-{dt.year}.json"
    )
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def flatten(obj, prefix=""):
    """Recursively flatten a nested dict into dot-separated keys."""
    out = {}
    for key, value in obj.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, full))
        elif isinstance(value, list):
            out[full] = value
        else:
            out[full] = value
    return out


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--date" and i + 1 < len(sys.argv[1:]):
            date_str = sys.argv[i + 2]

    # Download
    try:
        data = download_daily_json(date_str)
    except Exception as e:
        print(f"Failed: {e}")
        yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Trying {yesterday}")
        data = download_daily_json(yesterday)

    print(f"{len(data)} notices found")

    # Flatten all records and collect every field name
    all_fields = set()
    flat_records = []
    for record in data:
        flat = flatten(record)
        all_fields.update(flat.keys())
        flat_records.append(flat)

    # Save field list
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    fields_sorted = sorted(all_fields)
    fields_file = output_dir / "fields.txt"
    with open(fields_file, "w") as f:
        for field in fields_sorted:
            f.write(field + "\n")
    print(f"{len(fields_sorted)} unique fields saved to {fields_file}")

    # Print all fields
    print()
    print("All fields:")
    for field in fields_sorted:
        print(f"  {field}")

    # Print records with all non-null scalar values
    print()
    print(f"Notices ({len(flat_records)}):")
    for i, rec in enumerate(flat_records):
        print(f"\n--- {i + 1} ---")
        for key in sorted(rec.keys()):
            val = rec[key]
            if val is not None and val != "" and not isinstance(val, list):
                print(f"  {key}: {str(val)[:100]}")

    # Save raw JSON and flat records
    with open(output_dir / f"raw_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    with open(output_dir / f"flat_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(flat_records, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRaw JSON saved to data/raw_{date_str}.json")
    print(f"Flat records saved to data/flat_{date_str}.json")


if __name__ == "__main__":
    main()