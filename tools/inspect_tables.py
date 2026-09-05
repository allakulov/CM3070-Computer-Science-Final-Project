"""Probe what the Table Transformer detects on a procurement's scanned pages.

With ENABLE_TABLE_TRANSFORMER on, a scanned procurement produced no tables, yet the
model still rendered every page. Before deciding whether that path earns its place,
this answers the prior question: on the scanned PDFs, does the detection model find
nothing at all, or does it find a table whose structure then collapses to an empty
grid (which _cells_to_rows discards)? It runs only the geometry stages - the
detection model, then the structure model on each detected table - and reports the
boxes and row/column counts. It does NOT OCR any cell; the contents are not the
question here, and skipping OCR keeps the probe cheap.

The production threshold is high (TATR_THRESHOLD = 0.7), so --threshold lets you
lower it to see whether tables are being detected weakly and filtered out.

Run:
    python inspect_tables.py --id 124345
    python inspect_tables.py --id 124345 --threshold 0.3   # lower, to see near-misses
"""
from __future__ import annotations

import argparse
from pathlib import Path

from readers import (iter_container_files, read_plain, _pdf_page_images,
                     _load_tatr, MIN_USEFUL_CHARS, TATR_THRESHOLD)


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/*.zip
TABLE_LABELS = ("table", "table rotated")


# CORE

def detect(processor, model, image, threshold):
    """Run one Table Transformer model on an image; return [{label, score, box}].

    Same call as readers._detect_objects, but the threshold is a parameter so the
    probe can look below the production cut-off.
    """
    import torch
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    result = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes)[0]
    return [{"label": model.config.id2label[int(label)],
             "score": float(score),
             "box": [int(v) for v in box.tolist()]}
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"])]


def scanned_pdfs(eis_id):
    """Yield (name, page_images) for each PDF whose digital text is below the threshold."""
    for zip_path in sorted((DOWNLOADS_DIR / eis_id).glob("*.zip")):
        for name, data in iter_container_files(zip_path.read_bytes(), zip_path.name):
            if not name.lower().endswith(".pdf"):
                continue
            if len(read_plain(data, name).strip()) >= MIN_USEFUL_CHARS:
                continue                          # has a real text layer, not scanned
            yield name, _pdf_page_images(data)


def inspect(eis_id, threshold):
    """Report, per scanned page, what the detection and structure models find."""
    processor, detection, structure = _load_tatr()
    files = list(scanned_pdfs(eis_id))
    if not files:
        print(f"  no scanned PDFs found in {DOWNLOADS_DIR / eis_id}")
        return

    for name, images in files:
        print(f"\n{name} ({len(images)} pages, threshold {threshold})")
        found = 0
        for page_number, image in enumerate(images, 1):
            for obj in detect(processor, detection, image, threshold):
                if obj["label"] not in TABLE_LABELS:
                    continue
                found += 1
                cells = detect(processor, structure, image.crop(tuple(obj["box"])), threshold)
                n_rows = sum(c["label"] == "table row" for c in cells)
                n_cols = sum(c["label"] == "table column" for c in cells)
                if n_rows == 0:
                    note = "  -> 0 rows, discarded by _cells_to_rows"
                elif n_cols == 0:
                    note = "  -> 0 columns, cells would be empty"
                else:
                    note = ""
                print(f"    page {page_number}: {obj['label']} (score {obj['score']:.2f})"
                      f" -> {n_rows} rows x {n_cols} cols{note}")
        if found:
            print(f"    {found} table region(s) detected")
        else:
            print(f"    no tables detected on any page at threshold {threshold}"
                  f" (rerun with a lower --threshold to see near-misses)")


# RUNNER

def main():
    parser = argparse.ArgumentParser(
        description="Probe Table Transformer detection on a procurement's scanned pages.")
    parser.add_argument("--id", required=True,
                        help="procurement eis_id (folder under downloads/)")
    parser.add_argument("--threshold", type=float, default=TATR_THRESHOLD,
                        help=f"detection score threshold (default {TATR_THRESHOLD})")
    args = parser.parse_args()
    inspect(args.id, args.threshold)


if __name__ == "__main__":
    main()