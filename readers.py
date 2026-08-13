"""Reading layer: turn procurement files into text and tables.

This module owns everything about getting content out of files, kept separate
from the graph logic in extract_graph.py. It handles nested archives, several
file formats, an OCR fallback for scanned PDFs, and table extraction.

Two outputs per file:
  * text   - via read_text(), a sequence of attempts (TEXT_READERS): read_plain
             first, then read_ocr for scanned PDFs. The CPV pipeline uses this.
  * tables - via extract_tables(), one entry per table found. DOCX, XLSX, and
             digital PDFs use cheap library calls; scanned tables are recognised
             with RapidTable behind the ENABLE_TABLE_TRANSFORMER flag.

read_file() returns both of the above from one parse; read_text() and
extract_tables() remain for callers that want only one side.

The base readers (pdfplumber, python-docx, openpyxl) are imported at the top.
The optional OCR and Table Transformer paths pull in torch, which is slow to
import and large to install, so those libraries are imported inside the
functions that use them; when they are missing, OCR and table extraction report
it with an install hint and return nothing.
"""

from __future__ import annotations

import io
import re
import time
import zipfile

# Base readers, used on every run, are imported here as usual. The optional OCR
# (rapidocr, onnxruntime) and Table Transformer (transformers, torch) paths are
# imported inside the functions that use them, so the CPV pipeline starts quickly
# without them installed.
import docx
import openpyxl
import pdfplumber


# CONFIGURATION

CONTAINER_EXTS = (".zip", ".edoc")       # archive types to open instead of read as text
MAX_CONTAINER_DEPTH = 5                  # stop runaway recursion on nested archives
MIN_USEFUL_CHARS = 50                    # ignore files that yield almost no text

OCR_DPI = 200                            # resolution for rasterising scanned PDF pages

ENABLE_TABLE_TRANSFORMER = True
TATR_DETECTION_MODEL = "microsoft/table-transformer-detection"
TATR_STRUCTURE_MODEL = "microsoft/table-structure-recognition-v1.1-all"
TATR_THRESHOLD = 0.7                     # min confidence for a detected table/row/column
RAPIDOCR_MODEL_TYPE = "small"            # PP-OCRv6 tier for the page OCR ('small' or 'medium')


# ARCHIVES

def iter_container_files(data, container_name, depth=0):
    """Yield (filename, bytes) for every leaf file inside a ZIP or .edoc archive.

    Nested archives are opened recursively, so a ZIP inside a ZIP (or a document
    inside an .edoc) is unpacked until only real files remain. Bytes that are not
    a valid archive are yielded unchanged as a single leaf file.
    """
    if depth >= MAX_CONTAINER_DEPTH:
        return
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                payload = archive.read(info)
                if name.lower().endswith(CONTAINER_EXTS):
                    yield from iter_container_files(payload, name, depth + 1)
                else:
                    yield name, payload
    except zipfile.BadZipFile:
        yield container_name, data


# TEXT READERS
#
# Each reader takes (bytes, filename) and returns text, or "" if it cannot read
# the file. read_text tries them in order, so read_ocr only runs when read_plain
# came back empty (i.e. a scanned PDF).

def read_plain(data, name):
    """Extract text from PDF, DOCX, XLSX, or TXT bytes by file extension."""
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        if lower.endswith(".docx"):
            return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        if lower.endswith((".xlsx", ".xlsm")):
            workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            rows = []
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    cells = [str(c) for c in row if c is not None]
                    if cells:
                        rows.append("\t".join(cells))
            return "\n".join(rows)
        if lower.endswith((".txt", ".csv")):
            return data.decode("utf-8", errors="ignore")
    except Exception as error:
        print(f"    could not read {name}: {error}")
    return ""


def _pdf_page_images(data):
    """Render each PDF page to a PIL image, for OCR or table detection."""
    import fitz  # PyMuPDF
    from PIL import Image
    images = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=OCR_DPI)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


_rapidocr_reader = None


def _get_rapidocr_reader():
    """Build and cache the RapidOCR engine (PP-OCRv6, covers Latvian) once."""
    global _rapidocr_reader
    if _rapidocr_reader is None:
        from rapidocr import RapidOCR, ModelType, OCRVersion
        _rapidocr_reader = RapidOCR(params={"Rec.ocr_version": OCRVersion("PP-OCRv6"),
                                            "Rec.model_type": ModelType(RAPIDOCR_MODEL_TYPE)})
    return _rapidocr_reader


def _ocr_texts(reader, image):
    """Run RapidOCR on an image; return the list of recognised text pieces."""
    import numpy as np
    result = reader(np.array(image))
    texts = getattr(result, "txts", None)               # rapidocr v3 result object
    if texts:
        return list(texts)
    if isinstance(result, tuple) and result[0]:         # older (result, elapse) form
        return [row[1] for row in result[0]]
    return []


def _ocr_cell(reader, crop):
    """Read one table-cell image with RapidOCR; return its text (pieces joined)."""
    return " ".join(_ocr_texts(reader, crop))


def read_ocr(data, name, images=None):
    """Recover text from a scanned PDF with RapidOCR; non-PDFs return "".

    Args:
      data: The PDF's raw bytes.
      name: File name; must end in .pdf.
      images: Optional pre-rendered pages to reuse instead of rasterising again.

    Returns:
      The recognised text, or "" with an install hint when RapidOCR is missing.
    """
    if not name.lower().endswith(".pdf"):
        return ""
    try:
        reader = _get_rapidocr_reader()            # RapidOCR (PP-OCRv6)
        if images is None:
            images = _pdf_page_images(data)        # imports fitz (PyMuPDF)
        lines = []
        for image in images:
            lines.extend(_ocr_texts(reader, image))
        return "\n".join(lines)
    except ImportError as error:
        print(f"    skipping OCR for {name}: run 'pip install rapidocr onnxruntime pymupdf' to enable it ({error})")
        return ""
    except Exception as error:
        print(f"    OCR failed for {name}: {error}")
        return ""


TEXT_READERS = [read_plain, read_ocr]


def read_text(data, name):
    """Return text from the first reader that yields a usable amount."""
    for reader in TEXT_READERS:
        text = reader(data, name)
        if text and len(text.strip()) >= MIN_USEFUL_CHARS:
            return text
    return ""


# TABLE EXTRACTION
#
# Each returned table is {"source": filename, "rows": [[cell, ...], ...]}.
# Presence-checking is built in: a file with no tables yields an empty list.

def _tables_from_docx(data):
    """Read Word tables as lists of rows. Word exposes them directly."""
    document = docx.Document(io.BytesIO(data))
    tables = []
    for table in document.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if rows:
            tables.append(rows)
    return tables


def _tables_from_xlsx(data):
    """Treat each worksheet as one table of rows."""
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    tables = []
    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if c is None else str(c) for c in row]
            if any(cell.strip() for cell in cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _tables_from_pdf(data, name):
    """Digital PDFs: pdfplumber finds tables in the text/line layer (no ML).

    A PDF with no text layer is scanned; fall back to the Table Transformer
    models, but only when ENABLE_TABLE_TRANSFORMER is on.
    """
    tables = []
    has_text = False
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            if (page.extract_text() or "").strip():
                has_text = True
            for table in page.extract_tables():
                rows = [["" if c is None else str(c) for c in row] for row in table]
                if rows:
                    tables.append(rows)
    if not has_text and ENABLE_TABLE_TRANSFORMER:
        tables.extend(_tables_from_scanned_pdf(data))
    return tables


def extract_tables(data, name):
    """Return a list of tables found in one file; empty when there are none."""
    lower = name.lower()
    try:
        if lower.endswith(".docx"):
            rows_list = _tables_from_docx(data)
        elif lower.endswith((".xlsx", ".xlsm")):
            rows_list = _tables_from_xlsx(data)
        elif lower.endswith(".pdf"):
            rows_list = _tables_from_pdf(data, name)
        else:
            rows_list = []
    except Exception as error:
        print(f"    table extraction failed for {name}: {error}")
        return []
    return [{"source": name, "rows": rows} for rows in rows_list]


# TABLE TRANSFORMER
#
# Two DETR-based models. The pipeline uses only the detection model, to locate table
# regions that RapidTable then reconstructs; the structure model and _cells_to_rows
# stay for compare_tables.py.

_tatr_processor = None
_tatr_detection = None
_tatr_structure = None


def _load_tatr():
    """Lazily load and cache the detection and structure models plus processor."""
    global _tatr_processor, _tatr_detection, _tatr_structure
    if _tatr_detection is None:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        _tatr_processor = AutoImageProcessor.from_pretrained(TATR_DETECTION_MODEL)
        _tatr_detection = TableTransformerForObjectDetection.from_pretrained(TATR_DETECTION_MODEL)
        _tatr_structure = TableTransformerForObjectDetection.from_pretrained(TATR_STRUCTURE_MODEL)
    return _tatr_processor, _tatr_detection, _tatr_structure


def _detect_objects(processor, model, image):
    """Run a Table Transformer model on one image; return [{label, score, box}].

    Uses the processor's post_process_object_detection, which handles the
    box maths and drops the "no object" class for us (much less code than doing
    it by hand).
    """
    import torch
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    result = processor.post_process_object_detection(
        outputs, threshold=TATR_THRESHOLD, target_sizes=target_sizes
    )[0]
    objects = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        objects.append({
            "label": model.config.id2label[int(label)],
            "score": float(score),
            "box": [int(v) for v in box.tolist()],
        })
    return objects


def _cells_to_rows(structure, image):
    """Turn detected rows and columns into text rows by OCR-ing each cell with RapidOCR."""
    rows = sorted((o for o in structure if o["label"] == "table row"),
                  key=lambda o: o["box"][1])           # top to bottom
    cols = sorted((o for o in structure if o["label"] == "table column"),
                  key=lambda o: o["box"][0])           # left to right
    reader = _get_rapidocr_reader()
    table = []
    for row in rows:
        line = []
        for col in cols:
            # A cell is the intersection of one row band and one column band.
            cell_box = (col["box"][0], row["box"][1], col["box"][2], row["box"][3])
            line.append(_ocr_cell(reader, image.crop(cell_box)))
        table.append(line)
    return table


_rapidtable_engine = None


def _get_rapidtable_engine():
    """Build and cache the RapidTable engine (SLANet structure + RapidOCR cells) once."""
    global _rapidtable_engine
    if _rapidtable_engine is None:
        from rapid_table import RapidTable
        _rapidtable_engine = RapidTable()
    return _rapidtable_engine


def _html_to_rows(html_text):
    """Parse a RapidTable HTML table into a list of rows of cell text."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, re.S):
        cells = [re.sub(r"<[^>]+>", "", cell).strip()
                 for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if any(cells):
            rows.append(cells)
    return rows


def _mask_regions(images, boxes_per_page):
    """Copy each page with its table boxes painted white, so the full-page OCR skips them."""
    from PIL import ImageDraw
    out = []
    for image, boxes in zip(images, boxes_per_page):
        if boxes:
            image = image.copy()
            draw = ImageDraw.Draw(image)
            for box in boxes:
                draw.rectangle(box, fill="white")
        out.append(image)
    return out


def _tables_from_images(images):
    """Detect table regions with the Table Transformer and reconstruct them with RapidTable.

    Returns:
      (tables, table_boxes_per_page). A box is reported only when RapidTable returns a
      table for it, so the caller can mask captured regions from the full-page OCR while
      a failed region is left for the OCR.
    """
    import numpy as np
    processor, detection, _ = _load_tatr()            # detection only
    engine = _get_rapidtable_engine()
    tables, boxes_per_page = [], []
    for image in images:
        page_boxes = []
        for obj in _detect_objects(processor, detection, image):
            if obj["label"] not in ("table", "table rotated"):
                continue
            box = tuple(obj["box"])
            try:
                htmls = engine(np.asarray(image.crop(box))).pred_htmls
                rows = _html_to_rows(htmls[0]) if htmls else []
            except Exception as error:
                print(f"    table recognition failed on a region: {error}")
                rows = []
            if rows:
                tables.append(rows)
                page_boxes.append(box)                # mask only regions we captured
        boxes_per_page.append(page_boxes)
    return tables, boxes_per_page


def _tables_from_scanned_pdf(data):
    """Render a scanned PDF's pages and recognise tables on them (renders here)."""
    tables, _ = _tables_from_images(_pdf_page_images(data))
    return tables


# COMBINED READER

def _pdf_text_and_tables(data):
    """Read a digital PDF's text and pdfplumber tables; return (text, tables, has_text)."""
    text_parts, rows_list = [], []
    has_text = False
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                has_text = True
            text_parts.append(page_text)
            for table in page.extract_tables():
                rows = [["" if c is None else str(c) for c in row] for row in table]
                if rows:
                    rows_list.append(rows)
    return "\n".join(text_parts), rows_list, has_text


def _docx_text_and_tables(data):
    """Read a DOCX's paragraph text and its tables."""
    document = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs)
    rows_list = []
    for table in document.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if rows:
            rows_list.append(rows)
    return text, rows_list


def _xlsx_text_and_tables(data):
    """Read an XLSX's cell text and per-sheet tables in one pass (read-only sheets iterate once)."""
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    text_lines, rows_list = [], []
    for sheet in workbook.worksheets:
        sheet_rows = []
        for row in sheet.iter_rows(values_only=True):
            text_cells = [str(c) for c in row if c is not None]
            if text_cells:
                text_lines.append("\t".join(text_cells))
            table_cells = ["" if c is None else str(c) for c in row]
            if any(cell.strip() for cell in table_cells):
                sheet_rows.append(table_cells)
        if sheet_rows:
            rows_list.append(sheet_rows)
    return "\n".join(text_lines), rows_list


def read_file(data, name, info=None):
    """Read one file, returning its text and tables from a single parse.

    Args:
      data: The file's raw bytes.
      name: File name; its extension selects the reader.
      info: Optional dict, filled in for the caller to report on the read, with keys
        reader ("pdf"/"docx"/"xlsx"/"text"/"ocr", or None if unread), reason,
        parse_seconds and ocr_seconds.

    Returns:
      (text, tables). text is the plain reader's output once it clears MIN_USEFUL_CHARS,
      the OCR fallback for a scanned PDF, or "". tables holds one {"source", "rows"}
      entry per table.
    """
    if info is None:
        info = {}                 # local scratch when the caller passed none
    info["reader"] = None
    info["reason"] = ""
    info["parse_seconds"] = 0.0
    info["ocr_seconds"] = 0.0

    lower = name.lower()
    started = time.perf_counter()
    ocr_images = None             # scanned pages for the full-page OCR (table areas blanked)
    try:
        if lower.endswith(".pdf"):
            plain, rows_list, has_text = _pdf_text_and_tables(data)
            fmt = "pdf"
            if not has_text:
                page_images = _pdf_page_images(data)
                ocr_images = page_images
                if ENABLE_TABLE_TRANSFORMER:
                    file_tables, table_boxes = _tables_from_images(page_images)
                    rows_list.extend(file_tables)
                    # blank captured tables so the full-page OCR does not read them twice
                    ocr_images = _mask_regions(page_images, table_boxes)
        elif lower.endswith(".docx"):
            plain, rows_list = _docx_text_and_tables(data)
            fmt = "docx"
        elif lower.endswith((".xlsx", ".xlsm")):
            plain, rows_list = _xlsx_text_and_tables(data)
            fmt = "xlsx"
        elif lower.endswith((".txt", ".csv")):
            plain, rows_list = data.decode("utf-8", errors="ignore"), []
            fmt = "text"
        else:
            ext = name.lower().rsplit(".", 1)[-1] if "." in name else "?"
            info["reason"] = f"unsupported type (.{ext})"
            return "", []
    except Exception as error:
        info["reason"] = f"{type(error).__name__}: {error}"
        print(f"    could not read {name}: {error}")
        return "", []
    finally:
        info["parse_seconds"] = round(time.perf_counter() - started, 2)

    # Same threshold and OCR fallback as read_text: keep the plain text when it is
    # substantial, else try OCR (a no-op for non-PDFs), else treat the file as unread.
    # page_images is reused when present, so the scan is not rasterised a second time.
    if len(plain.strip()) >= MIN_USEFUL_CHARS:
        info["reader"] = fmt
        text = plain
    else:
        started = time.perf_counter()
        ocr = read_ocr(data, name, ocr_images)
        info["ocr_seconds"] = round(time.perf_counter() - started, 2)
        if len(ocr.strip()) >= MIN_USEFUL_CHARS:
            info["reader"] = "ocr"
            text = ocr
        else:
            info["reason"] = ("no usable text (after OCR attempt)" if fmt == "pdf"
                              else f"under {MIN_USEFUL_CHARS}-char threshold")
            text = ""

    tables = [{"source": name, "rows": rows} for rows in rows_list]
    return text, tables