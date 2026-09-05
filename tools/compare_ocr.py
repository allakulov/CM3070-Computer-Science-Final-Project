"""Compare OCR engines on a procurement's scanned PDFs.

We saw EasyOCR mangle Latvian diacritics and linearise a table, so the choice of
OCR model should rest on evidence rather than the one engine that happened to be
wired in first. This runs several engines on the SAME rasterised pages and reports
characters and seconds per file per engine. It deliberately computes no accuracy
score: there is no ground truth here, so it writes each engine's text to disk and
you judge quality by diffing those files.

The engines:
  * easyocr   - the deep-learning reader readers.py already uses (CRAFT + CRNN).
  * tesseract - the classic engine (LSTM), via pytesseract, using the Latvian model.
  * rapidocr  - PP-OCR on ONNX, configured with the Latin recognition model.

It reads the same downloads/{eis_id}/*.zip a normal run reads, keeps the PDFs whose
digital text layer is below MIN_USEFUL_CHARS (the scanned ones), rasterises each
once via readers._pdf_page_images, and runs every requested engine on those pages.
A missing engine is skipped with a note rather than stopping the run.

Install:
    pip install pytesseract rapidocr onnxruntime      # easyocr is already used by readers
    # Tesseract also needs the system binary and Latvian data:
    #   Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-lav
    #   macOS:         brew install tesseract tesseract-lang

Run:
    python compare_ocr.py --id 470663
    python compare_ocr.py --id 470663 --engines easyocr tesseract
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from readers import iter_container_files, read_plain, _pdf_page_images, MIN_USEFUL_CHARS


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/*.zip
COMPARE_DIR = Path("compare_ocr")        # output: compare_ocr/{eis_id}/{file}.{engine}.txt
TESSERACT_LANG = "lav"                   # Latvian traineddata (lav) in every tessdata build
RAPIDOCR_MODEL_TYPE = "small"            # PP-OCRv6 tier: 'small' (fast) or 'medium' (most accurate)


# ENGINES
# Each engine is lazy-imported so a missing one is skipped, not fatal. Each takes
# the list of PIL page images and returns the recognised text as one string.

_easyocr_reader = None


def ocr_easyocr(images):
    """EasyOCR (CRAFT + CRNN), Latvian + English."""
    import numpy as np
    import easyocr
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["lv", "en"])
    lines = []
    for image in images:
        for _, text, _ in _easyocr_reader.readtext(np.array(image)):
            lines.append(text)
    return "\n".join(lines)


def ocr_tesseract(images):
    """Tesseract via pytesseract, using the Latvian model."""
    import pytesseract
    return "\n".join(pytesseract.image_to_string(image, lang=TESSERACT_LANG)
                     for image in images)


_rapidocr_engine = None


def ocr_rapidocr(images):
    """RapidOCR with the PP-OCRv6 unified multilingual recogniser.

    PP-OCRv6's dictionary was extended with ~200 diacritical characters and covers
    Latvian, so no lang_type is set (v6 is one model for all its languages and
    ignores it). The earlier PP-OCRv5 'latin' model dropped a/e/i/g/k/l/n with
    macrons and cedillas entirely, which is why a v5 config fails on Latvian.
    """
    import numpy as np
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import RapidOCR, ModelType, OCRVersion
        # Enum(value) resolves by value, so this is robust to the member names.
        _rapidocr_engine = RapidOCR(params={"Rec.ocr_version": OCRVersion("PP-OCRv6"),
                                            "Rec.model_type": ModelType(RAPIDOCR_MODEL_TYPE)})
    lines = []
    for image in images:
        result = _rapidocr_engine(np.array(image))
        texts = getattr(result, "txts", None)               # rapidocr v3 result object
        if texts:
            lines.extend(texts)
        elif isinstance(result, tuple) and result[0]:       # older (result, elapse) form
            lines.extend(row[1] for row in result[0])
    return "\n".join(lines)


ENGINES = {"easyocr": ocr_easyocr, "tesseract": ocr_tesseract, "rapidocr": ocr_rapidocr}


# CORE

def scanned_pdfs(eis_id):
    """Yield (name, page_images) for each PDF whose digital text is below the threshold."""
    for zip_path in sorted((DOWNLOADS_DIR / eis_id).glob("*.zip")):
        for name, data in iter_container_files(zip_path.read_bytes(), zip_path.name):
            if not name.lower().endswith(".pdf"):
                continue
            if len(read_plain(data, name).strip()) >= MIN_USEFUL_CHARS:
                continue                          # has a real text layer, not scanned
            yield name, _pdf_page_images(data)


def compare(eis_id, engines):
    """Run each engine on every scanned PDF; print a table and write the texts out."""
    out_dir = COMPARE_DIR / eis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(scanned_pdfs(eis_id))
    if not files:
        print(f"  no scanned PDFs found in {DOWNLOADS_DIR / eis_id}")
        return

    for name, images in files:
        print(f"\n{name} ({len(images)} pages)")
        print(f"    {'engine':<10} {'chars':>8} {'seconds':>9}")
        stem = Path(name).name                    # flatten any nested archive path
        for engine in engines:
            try:
                started = time.perf_counter()
                text = ENGINES[engine](images)
                seconds = round(time.perf_counter() - started, 2)
            except ImportError as error:
                print(f"    {engine:<10} skipped: not installed ({error})")
                continue
            except Exception as error:
                print(f"    {engine:<10} failed: {error}")
                continue
            (out_dir / f"{stem}.{engine}.txt").write_text(text, encoding="utf-8")
            print(f"    {engine:<10} {len(text):>8,} {seconds:>9}")

    print(f"\n  wrote texts to {out_dir}/  (diff them to compare quality)")


# RUNNER

def main():
    parser = argparse.ArgumentParser(
        description="Compare OCR engines on a procurement's scanned PDFs.")
    parser.add_argument("--id", required=True,
                        help="procurement eis_id (folder under downloads/)")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES),
                        help="which engines to run (default: all)")
    args = parser.parse_args()
    compare(args.id, args.engines)


if __name__ == "__main__":
    main()