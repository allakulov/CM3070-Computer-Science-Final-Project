"""Extract the main CPV code from each procurement's documents.

This is a LangGraph pipeline. For one procurement it reads every document,
splits the combined text into overlapping chunks, asks a local LLM (Ollama)
for the main CPV code in each chunk, then merges the chunk results into a
single answer saved to ``extracted/{eis_id}.json``.

The graph contains a cycle: the ``extract_chunk`` node runs once per chunk,
looping back to itself until every chunk is processed, then continues to
``merge``. The ``partials`` list uses an ``operator.add`` reducer so each loop
iteration appends to it instead of overwriting it.

Validation is deliberately NOT done here. A separate script compares the saved
results against the structured open data, so this pipeline never sees the
answer key during extraction.

Install:
    pip install langgraph langchain-ollama pydantic
    pip install pdfplumber python-docx openpyxl
    ollama pull mistral-small

Run:
    python extract_graph.py                  # every procurement in downloads/
    python extract_graph.py --eis-id 123450  # just one
"""

from __future__ import annotations

import argparse
import io
import json
import operator
import zipfile
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/*.zip
OUTPUT_DIR = Path("extracted")           # output: extracted/{eis_id}.json

EXTRACTION_MODEL = "mistral-small"       # any tag listed by `ollama list`
OLLAMA_NUM_CTX = 4096                    # Ollama's default of 2048 is too small

CHUNK_SIZE = 3000                        # characters per chunk (about 1000 tokens)
CHUNK_OVERLAP = 200                      # shared characters between neighbouring chunks

# File types that are really archives and should be opened, not read as text.
# ".edoc" is the Latvian e-signed document format: a ZIP holding the real
# document plus signature files.
CONTAINER_EXTS = (".zip", ".edoc")
MAX_CONTAINER_DEPTH = 5                  # stop runaway recursion on nested archives

MIN_USEFUL_CHARS = 50                    # ignore files that yield almost no text


# EXTRACTION SCHEMA
#
# The class name, its docstring, and each field description are sent to the
# model as instructions, so they are written for the model to read.

class CpvCode(BaseModel):
    """Main CPV classification code of a procurement."""

    found: bool = Field(
        description="True only if a CPV code appears in this chunk."
    )
    cpv_code: Optional[str] = Field(
        None, description='The main CPV code, formatted like "71220000-6".'
    )
    description: Optional[str] = Field(
        None, description="What the code refers to, if stated (goods, services, or works)."
    )
    evidence: Optional[str] = Field(
        None, description="A short verbatim snippet from the chunk showing the code."
    )


EXTRACTION_PROMPT = """You are reading one chunk of a Latvian public procurement \
document. Find the main CPV code.

CPV (Common Procurement Vocabulary) codes classify what is being procured. A code \
is eight digits, a hyphen, then one check digit, for example "71220000-6". The \
Latvian label is "CPV kods". Return only the single main code. If this chunk has no \
CPV code, set found to false. Do not guess.

CHUNK TEXT:
{text}
"""


# MODEL
#
# Built once and reused for every chunk. Constructing ChatOllama does not open a
# connection, so importing this module without Ollama running is fine.

chat_model = ChatOllama(model=EXTRACTION_MODEL, temperature=0, num_ctx=OLLAMA_NUM_CTX)
extractor = chat_model.with_structured_output(CpvCode)


# FILE READERS
#
# Each reader takes (bytes, filename) and returns text, or "" if it cannot read
# the file. READER_PIPELINE is the ordered list of attempts. Today it holds one
# reader; OCR and table readers will be appended later without other changes.

def read_plain(data, name):
    """Extract text from PDF, DOCX, XLSX, or TXT bytes by file extension."""
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        if lower.endswith(".docx"):
            import docx
            return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        if lower.endswith((".xlsx", ".xlsm")):
            import openpyxl
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


READER_PIPELINE = [read_plain]


def read_file(data, name):
    """Return text from the first reader that produces a usable result."""
    for reader in READER_PIPELINE:
        text = reader(data, name)
        if text and len(text.strip()) >= MIN_USEFUL_CHARS:
            return text
    return ""


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
                    # "yield from" passes through every item from the inner archive.
                    yield from iter_container_files(payload, name, depth + 1)
                else:
                    yield name, payload
    except zipfile.BadZipFile:
        yield container_name, data


# CHUNKING

def split_into_chunks(text, size, overlap):
    """Split text into overlapping fixed-size chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# GRAPH STATE
#
# A single dict flows through every node. Nodes return only the keys they change,
# and LangGraph merges those into the state. "partials" is special: the
# operator.add reducer appends each node's list instead of replacing it, which is
# what lets the chunk loop accumulate results.

class State(TypedDict):
    eis_id: str
    source_files: list[str]
    documents_text: str
    chunks: list[str]
    current_chunk_idx: int
    partials: Annotated[list[dict], operator.add]
    final: Optional[dict]


# GRAPH NODES

def load_documents(state):
    """Read every leaf file in the procurement folder into one text blob."""
    folder = DOWNLOADS_DIR / state["eis_id"]
    texts = []
    file_names = []
    unread = []

    for zip_path in sorted(folder.glob("*.zip")):
        try:
            zip_bytes = zip_path.read_bytes()
        except OSError as error:
            print(f"    cannot read {zip_path.name}: {error}")
            continue

        for name, data in iter_container_files(zip_bytes, zip_path.name):
            text = read_file(data, name)
            if text:
                # Label each file so chunk boundaries keep some source context.
                texts.append(f"Source file: {name}\n{text}")
                file_names.append(name)
            else:
                unread.append(name)

    blob = "\n\n".join(texts)
    message = f"  load_documents: {len(blob):,} chars from {len(file_names)} files"
    if unread:
        message += f" ({len(unread)} unreadable, e.g. {unread[0]!r})"
    print(message)
    return {"documents_text": blob, "source_files": file_names}


def chunk_text(state):
    """Split the loaded text into overlapping chunks."""
    chunks = split_into_chunks(state["documents_text"], CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  chunk_text: {len(chunks)} chunks of up to {CHUNK_SIZE} chars")
    return {"chunks": chunks, "current_chunk_idx": 0}


def extract_chunk(state):
    """Extract the main CPV code from the current chunk and append the result."""
    idx = state["current_chunk_idx"]
    chunks = state["chunks"]
    prompt = EXTRACTION_PROMPT.format(text=chunks[idx])

    try:
        result = extractor.invoke(prompt)
        partial = result.model_dump()
    except Exception as error:
        # One bad chunk should not stop the whole procurement.
        print(f"    chunk {idx + 1}: extraction error: {error}")
        partial = {"found": False, "cpv_code": None, "description": None, "evidence": None}

    if partial.get("found") and partial.get("cpv_code"):
        status = f"found {partial['cpv_code']}"
    else:
        status = "no code"
    print(f"    chunk {idx + 1}/{len(chunks)}: {status}")

    # The single-item list is appended to state["partials"] by the reducer.
    return {"partials": [partial], "current_chunk_idx": idx + 1}


def should_continue_chunking(state):
    """Return the next node: keep looping if chunks remain, else merge."""
    if state["current_chunk_idx"] < len(state["chunks"]):
        return "extract_chunk"
    return "merge"


def merge(state):
    """Combine the per-chunk results into one main CPV code.

    Take the first chunk that found a code as the main result, but also record
    every distinct code seen across chunks so additional codes are visible for
    later work.
    """
    found = [p for p in state["partials"] if p.get("found") and p.get("cpv_code")]

    if not found:
        final = {"found": False, "cpv_code": None, "description": None, "codes_seen": []}
    else:
        first = found[0]
        final = {
            "found": True,
            "cpv_code": first["cpv_code"],
            "description": first.get("description"),
            "evidence": first.get("evidence"),
            "codes_seen": sorted({p["cpv_code"] for p in found}),
        }

    print(f"  merge: main code {final['cpv_code']} (from {len(found)} chunks)")
    return {"final": final}


# GRAPH ASSEMBLY

def build_graph():
    """Build and compile the extraction graph."""
    graph = StateGraph(State)
    graph.add_node("load_documents", load_documents)
    graph.add_node("chunk_text", chunk_text)
    graph.add_node("extract_chunk", extract_chunk)
    graph.add_node("merge", merge)

    graph.add_edge(START, "load_documents")
    graph.add_edge("load_documents", "chunk_text")
    graph.add_edge("chunk_text", "extract_chunk")
    # The cycle: after each chunk, decide whether to loop back or move on.
    graph.add_conditional_edges(
        "extract_chunk",
        should_continue_chunking,
        {"extract_chunk": "extract_chunk", "merge": "merge"},
    )
    graph.add_edge("merge", END)

    return graph.compile()


def save_graph_image(graph, path="graph.png"):
    """Save the graph as a PNG image.

    draw_mermaid_png renders through the mermaid.ink web service, so it needs
    internet access. If rendering fails, write the Mermaid source text instead.
    """
    try:
        Path(path).write_bytes(graph.get_graph().draw_mermaid_png())
        print(f"graph image saved to {path}")
    except Exception as error:
        fallback = Path(path).with_suffix(".mmd")
        fallback.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"could not render PNG ({error}); wrote Mermaid text to {fallback}")


# RUNNER

def initial_state(eis_id):
    """Return a fresh state for one procurement."""
    return {
        "eis_id": eis_id,
        "source_files": [],
        "documents_text": "",
        "chunks": [],
        "current_chunk_idx": 0,
        "partials": [],
        "final": None,
    }


def process_one(graph, eis_id):
    """Run the graph for one procurement and save the result to disk."""
    print(f"id: {eis_id}")
    try:
        state = graph.invoke(initial_state(eis_id))
    except Exception as error:
        print(f"  pipeline failed: {error}")
        return

    result = {
        "eis_id": eis_id,
        "source_files": state.get("source_files", []),
        "n_chunks": len(state.get("chunks", [])),
        "extracted": state.get("final"),
        "partials": state.get("partials", []),   # kept for debugging
    }
    out_path = OUTPUT_DIR / f"{eis_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {out_path}\n")


def main():
    """Run extraction over the selected procurements."""
    parser = argparse.ArgumentParser(description="Extract main CPV codes from procurements.")
    parser.add_argument("--eis-id", help="Process only this procurement id")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.eis_id:
        eis_ids = [args.eis_id]
    else:
        eis_ids = sorted(p.name for p in DOWNLOADS_DIR.iterdir() if p.is_dir())

    if not eis_ids:
        print(f"No procurements found in {DOWNLOADS_DIR}")
        return

    graph = build_graph()
    save_graph_image(graph)
    print()

    for eis_id in eis_ids:
        if (DOWNLOADS_DIR / eis_id).is_dir():
            process_one(graph, eis_id)
        else:
            print(f"skipping {eis_id}: no folder in {DOWNLOADS_DIR}")


if __name__ == "__main__":
    main()


# ROADMAP
#
# 1. additionalCpvType: procurements often list extra CPV codes beyond the main
#    one, and these are frequently MISSING from the structured open data, so
#    there is no easy answer key to validate against. This likely needs a
#    self-checking agent (a "reflexion" loop: extract, critique its own output
#    against the document, then retry) rather than a single pass. The merge node
#    already records "codes_seen" as a starting point.
#
# 2. Validation: a separate script reads extracted/{eis_id}.json, looks up the
#    structured CPV value, and reports match rates. Kept out of this pipeline so
#    extraction never sees the answer key.
#
# 3. More readers: append OCR (for scanned PDFs) and a .doc reader to
#    READER_PIPELINE. read_file already tries readers in order.
#
# 4. Per-chunk classification: label each chunk before extraction and use a
#    conditional edge to skip boilerplate or route chunk types to specialised
#    agents. This is where multi-agent orchestration enters the graph.