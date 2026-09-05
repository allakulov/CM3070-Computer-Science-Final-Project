"""Compare the current scanned-table path against SLANet (RapidTable) on the same crops.

Both reconstructions run on identical table regions - the ones the Table Transformer
detector already finds - so their grids can be compared directly:
  A) current    - TATR structure recognition + per-cell RapidOCR (readers._cells_to_rows)
  B) rapidtable - SLANet-plus structure (ONNX) + RapidOCR, via the rapid_table package
  C) docling    - IBM TableFormer via Docling: an all-in-one recogniser that does
                  structure and cell text together, with no separate OCR step

SLANet is the table-structure model from PP-StructureV2 (Li et al., 2022,
arXiv:2210.05391), an image-to-HTML structure recogniser; RapidTable (RapidAI,
Apache-2.0) is its ONNX inference wrapper and reuses the RapidOCR engine already used
here for cell text. For each detected table, one self-contained HTML page is written to
compare_tables/{id}/ rendering all three reconstructions side by side for eyeballing -
no accuracy score is computed (there is no ground truth).

Install:
    pip install rapid_table          # for B; rapidocr and the TATR deps are already installed
    pip install docling              # for C (first run downloads the TableFormer models)

Run:
    python compare_tables.py --id 124345
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from readers import (iter_container_files, read_plain, _pdf_page_images, _load_tatr,
                     _detect_objects, _cells_to_rows, MIN_USEFUL_CHARS)


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/*.zip
COMPARE_DIR = Path("compare_tables")     # output: compare_tables/{eis_id}/{tag}.{engine}.html
TABLE_LABELS = ("table", "table rotated")


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


def _rows_to_html(rows):
    """Wrap the current path's list-of-rows reconstruction as an HTML table."""
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                   for row in rows)
    return f"<table>{body}</table>"


def _table_only(markup):
    """Keep just the <table>...</table> so each engine's output embeds cleanly."""
    start, end = markup.find("<table"), markup.rfind("</table>")
    return markup[start:end + len("</table>")] if start != -1 and end != -1 else markup


def _side_by_side(tag, panels):
    """One self-contained page rendering each reconstruction in its own column."""
    columns = "".join("<div class='col'><h2>" + title + "</h2>" + _table_only(markup) + "</div>"
                      for title, markup in panels)
    return (
        "<!doctype html><meta charset='utf-8'><title>" + tag + "</title>"
        "<style>body{font:14px sans-serif;margin:16px}"
        ".cols{display:flex;gap:24px;align-items:flex-start}.col{flex:1;min-width:0}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #999;padding:4px;vertical-align:top}"
        "h2{font-size:14px;background:#eee;padding:6px;margin:0 0 8px}</style>"
        "<h1>" + tag + "</h1><div class='cols'>" + columns + "</div>")


def _current_html(processor, structure_model, crop):
    """Current path: TATR structure recognition + per-cell RapidOCR."""
    cells = _detect_objects(processor, structure_model, crop)
    return _rows_to_html(_cells_to_rows(cells, crop))


_rapid_engine = None


def _rapidtable_html(crop):
    """SLANet-plus structure + RapidOCR for one table crop, via rapid_table."""
    import numpy as np
    global _rapid_engine
    if _rapid_engine is None:
        from rapid_table import RapidTable
        _rapid_engine = RapidTable()             # SLANet-plus (ONNX); auto-uses installed RapidOCR
    out = _rapid_engine(np.asarray(crop))
    if hasattr(out, "pred_htmls"):               # rapid_table v2+ returns a list of tables
        return out.pred_htmls[0] if out.pred_htmls else ""
    if hasattr(out, "pred_html"):                # older single-table field
        return out.pred_html
    if isinstance(out, tuple):                   # oldest (html, elapse) form
        return out[0]
    return str(out)


_docling_converter = None


def _docling_html(crop):
    """TableFormer via Docling: an all-in-one recogniser (structure + built-in OCR).

    Docling detects and reconstructs the table itself and supplies the cell text with
    its own OCR, so there is no separate OCR step here. The crop is written to a temp
    image because Docling's converter takes a file. NB: Docling's API shifts across
    versions - export_to_html gained a `doc` argument - so both forms are tried.
    """
    import os
    import tempfile
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        crop.save(handle.name)
        handle.close()
        doc = _docling_converter.convert(handle.name).document
        tables = getattr(doc, "tables", [])
        if not tables:
            return "<!-- docling: no table found -->"
        try:
            return tables[0].export_to_html(doc=doc)     # newer docling
        except TypeError:
            return tables[0].export_to_html()            # older docling
    finally:
        os.unlink(handle.name)


def _safe(engine, crop, hint):
    """Run an engine, returning an HTML comment instead of raising if it is unavailable."""
    try:
        return engine(crop)
    except ImportError as error:
        return f"<!-- not installed: {hint} ({error}) -->"
    except Exception as error:
        return f"<!-- failed: {error} -->"


def compare(eis_id):
    """Run both table reconstructions on every detected table region and write the HTML."""
    out_dir = COMPARE_DIR / eis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    processor, detection, structure = _load_tatr()

    files = list(scanned_pdfs(eis_id))
    if not files:
        print(f"  no scanned PDFs found in {DOWNLOADS_DIR / eis_id}")
        return

    total = 0
    for name, images in files:
        stem = Path(name).name
        for page_number, image in enumerate(images, 1):
            tables = [o for o in _detect_objects(processor, detection, image)
                      if o["label"] in TABLE_LABELS]
            for table_index, obj in enumerate(tables, 1):
                crop = image.crop(tuple(obj["box"]))
                tag = f"{stem}_p{page_number}_t{table_index}"
                panels = [
                    ("Current \u2014 TATR + RapidOCR", _current_html(processor, structure, crop)),
                    ("RapidTable \u2014 SLANet + RapidOCR", _safe(_rapidtable_html, crop, "pip install rapid_table")),
                    ("Docling \u2014 TableFormer", _safe(_docling_html, crop, "pip install docling")),
                ]
                (out_dir / f"{tag}.html").write_text(_side_by_side(tag, panels), encoding="utf-8")
                total += 1
                print(f"    {tag}: wrote side-by-side")

    print(f"\n  {total} table region(s) compared -> {out_dir}/"
          f"  (open each {{tag}}.html in a browser to see all three side by side)")


# RUNNER

def main():
    parser = argparse.ArgumentParser(
        description="Compare the current table path against SLANet (RapidTable).")
    parser.add_argument("--id", required=True,
                        help="procurement eis_id (folder under downloads/)")
    args = parser.parse_args()
    compare(args.id)


if __name__ == "__main__":
    main()