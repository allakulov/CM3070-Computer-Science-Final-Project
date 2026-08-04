"""Reading layer: turn procurement files into text and tables.

This module owns everything about getting content out of files, kept separate
from the graph logic in extract_graph.py. It handles nested archives, several
file formats, an OCR fallback for scanned PDFs, and table extraction.

Two outputs per file:
  * text   - via read_text(), a sequence of attempts (TEXT_READERS): read_plain
             first, then read_ocr for scanned PDFs. The CPV pipeline uses this.
  * tables - via extract_tables(), one entry per table found. DOCX, XLSX, and
             digital PDFs use cheap library calls; scanned PDFs use the Table
             Transformer models, which are OFF by default because they are heavy.

read_file() returns both of the above from a single parse, so the extraction loop
reads each file once instead of twice; read_text() and extract_tables() are kept
unchanged for callers that want only one side.

The base readers (pdfplumber, python-docx, openpyxl) are imported at the top.
The optional OCR and Table Transformer paths pull in torch, which is slow to
import and large to install, so those libraries are imported inside the
functions that use them; when they are missing, OCR and table extraction report
it with an install hint and return nothing.
"""

from __future__ import annotations

import io
import zipfile

# Base readers, used on every run, are imported here as usual. The optional OCR
# and Table Transformer paths pull in torch (via easyocr / transformers), which
# is slow to import and large to install, so those are imported inside the
# functions that use them -- the CPV pipeline then runs, and starts quickly,
# without them installed.
import docx
import openpyxl
import pdfplumber


# CONFIGURATION

CONTAINER_EXTS = (".zip", ".edoc")       # archive types to open instead of read as text
MAX_CONTAINER_DEPTH = 5                  # stop runaway recursion on nested archives
MIN_USEFUL_CHARS = 50                    # ignore files that yield almost no text

OCR_LANGUAGES = ["lv", "en"]             # EasyOCR codes; Latvian + English (both Latin script)
OCR_DPI = 200                            # resolution for rasterising scanned PDF pages

# The Table Transformer path is heavy (two deep-learning models + OCR per cell)
# and only needed for SCANNED tables, so it is off by default. Turn it on when you
# start extracting structured fields from scanned documents.
ENABLE_TABLE_TRANSFORMER = True
TATR_DETECTION_MODEL = "microsoft/table-transformer-detection"
TATR_STRUCTURE_MODEL = "microsoft/table-structure-recognition-v1.1-all"
TATR_THRESHOLD = 0.7                     # min confidence for a detected table/row/column


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


_ocr_reader = None


def _get_ocr_reader():
    """Build and cache the EasyOCR reader once (it loads models on first use)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES)
    return _ocr_reader


def read_ocr(data, name):
    """Recover text from a scanned PDF by rasterising its pages and running OCR.

    Only attempts PDFs. If the OCR libraries are not installed it prints an
    install hint and returns "", so it stays an optional fallback after read_plain.
    """
    if not name.lower().endswith(".pdf"):
        return ""
    try:
        import numpy as np
        reader = _get_ocr_reader()                 # imports easyocr (and torch)
        lines = []
        for image in _pdf_page_images(data):       # imports fitz (PyMuPDF)
            for _, text, _ in reader.readtext(np.array(image)):
                lines.append(text)
        return "\n".join(lines)
    except ImportError as error:
        print(f"    skipping OCR for {name}: run 'pip install easyocr pymupdf' to enable it ({error})")
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


# TABLE TRANSFORMER (optional, scanned tables only)
#
# Two DETR-based models: one detects table regions, one recognises a table's rows
# and columns. Neither does OCR, so we read each cell's text with EasyOCR. This
# path is untested without the model weights downloaded; it is off by default.

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
    """Turn detected rows and columns into text rows by OCR-ing each cell."""
    import numpy as np
    rows = sorted((o for o in structure if o["label"] == "table row"),
                  key=lambda o: o["box"][1])           # top to bottom
    cols = sorted((o for o in structure if o["label"] == "table column"),
                  key=lambda o: o["box"][0])           # left to right
    reader = _get_ocr_reader()
    table = []
    for row in rows:
        line = []
        for col in cols:
            # A cell is the intersection of one row band and one column band.
            cell_box = (col["box"][0], row["box"][1], col["box"][2], row["box"][3])
            crop = np.array(image.crop(cell_box))
            text = " ".join(t for _, t, _ in reader.readtext(crop))
            line.append(text)
        table.append(line)
    return table


def _tables_from_scanned_pdf(data):
    """Detect tables on each scanned page and reconstruct their rows."""
    processor, detection, structure = _load_tatr()
    tables = []
    for image in _pdf_page_images(data):
        for obj in _detect_objects(processor, detection, image):
            if obj["label"] not in ("table", "table rotated"):
                continue
            crop = image.crop(tuple(obj["box"]))
            cells = _detect_objects(processor, structure, crop)
            rows = _cells_to_rows(cells, crop)
            if rows:
                tables.append(rows)
    return tables


# COMBINED READER (single parse per file)
#
# read_file() opens each file once and returns BOTH its text and its tables, so
# the reading loop no longer parses every file twice (once via read_text, once via
# extract_tables). Those functions and the per-format helpers above are left
# untouched for callers that use them directly; read_file reuses read_ocr and
# _tables_from_scanned_pdf, so the scanned fallbacks are shared. The one
# consequence of a single open: a hard parse error now yields empty text and empty
# tables together (one message), rather than each half failing independently.

def _pdf_text_and_tables(data):
    """Read a digital PDF's text and tables in one pass.

    Mirrors read_plain (the PDF branch) and _tables_from_pdf, but opens the file
    once: the page loop collects the text layer and the tables together, and the
    same pass decides whether the PDF is scanned (which drives the Table Transformer).
    """
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
    if not has_text and ENABLE_TABLE_TRANSFORMER:
        rows_list.extend(_tables_from_scanned_pdf(data))
    return "\n".join(text_parts), rows_list


def _docx_text_and_tables(data):
    """Read a DOCX's paragraph text and its tables from a single open."""
    document = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs)
    rows_list = []
    for table in document.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if rows:
            rows_list.append(rows)
    return text, rows_list


def _xlsx_text_and_tables(data):
    """Read an XLSX's cell text and per-sheet tables from a single open.

    The workbook is iterated once; each row becomes both the tab-joined text line
    read_plain produced and the full-width row _tables_from_xlsx kept.
    """
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


def read_file(data, name):
    """Open one file once and return (text, tables).

    Same outputs as read_text(data, name) and extract_tables(data, name) called
    separately, but the file is parsed a single time. text is the plain reader's
    output when it clears MIN_USEFUL_CHARS, otherwise the OCR fallback (scanned
    PDFs only), otherwise "". tables is one entry per table found, each shaped
    {"source": name, "rows": [[cell, ...], ...]}.
    """
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            plain, rows_list = _pdf_text_and_tables(data)
        elif lower.endswith(".docx"):
            plain, rows_list = _docx_text_and_tables(data)
        elif lower.endswith((".xlsx", ".xlsm")):
            plain, rows_list = _xlsx_text_and_tables(data)
        elif lower.endswith((".txt", ".csv")):
            plain, rows_list = data.decode("utf-8", errors="ignore"), []
        else:
            plain, rows_list = "", []
    except Exception as error:
        print(f"    could not read {name}: {error}")
        return "", []

    # Same threshold and OCR fallback as read_text: keep the plain text when it is
    # substantial, else try OCR (a no-op for non-PDFs), else treat the file as unread.
    if len(plain.strip()) >= MIN_USEFUL_CHARS:
        text = plain
    else:
        ocr = read_ocr(data, name)
        text = ocr if len(ocr.strip()) >= MIN_USEFUL_CHARS else ""

    tables = [{"source": name, "rows": rows} for rows in rows_list]
    return text, tables