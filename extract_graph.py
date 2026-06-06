"""Extract the main and additional CPV codes from a procurement's documents.

This is a LangGraph pipeline. For one procurement it reads every document, uses
a regex to find every CPV code together with its surrounding text, then asks a
local LLM (Ollama) to decide which code is the main one and which are additional.
A reflexion step lets the model critique its own answer and try again.

Why a regex first, then an LLM: the regex guarantees we catch every code and
gives us the evidence text for free, so the LLM only has to make a judgement over
a short candidate list instead of scanning the whole document. Because the model
never sees the full text, no chunking is needed for this field.

Pydantic does the type checking. CpvClassification validates the model's output
(codes must look like CPV codes; the main code is kept out of the additional
list), and Critique uses a Literal to limit the verdict to "accept" or "revise".

Validation is deliberately NOT done here. A separate script compares the saved
results against the structured open data, so this pipeline never sees the answer
key during extraction.

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
import re
import zipfile
from pathlib import Path
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/*.zip
OUTPUT_DIR = Path("extracted")           # output: extracted/{eis_id}.json

EXTRACTION_MODEL = "mistral-small"       # any tag listed by `ollama list`
OLLAMA_NUM_CTX = 4096                    # Ollama's default of 2048 is too small

CONTEXT_CHARS = 200                      # characters of context kept on each side of a code
MAX_CLASSIFY_ATTEMPTS = 3                # cap on the reflexion loop

CONTAINER_EXTS = (".zip", ".edoc")       # archive types to open instead of read as text
MAX_CONTAINER_DEPTH = 5                  # stop runaway recursion on nested archives
MIN_USEFUL_CHARS = 50                    # ignore files that yield almost no text

# A CPV code is eight digits, a hyphen, then one check digit, e.g. 71220000-6.
# The lookarounds stop us matching part of a longer run of digits.
CPV_REGEX = r"\d{8}-\d"
CPV_PATTERN = re.compile(r"(?<!\d)" + CPV_REGEX + r"(?!\d)")


def looks_like_cpv(code):
    """Return True if the string is exactly a CPV code."""
    return re.fullmatch(CPV_REGEX, code) is not None


# EXTRACTION SCHEMAS
#
# Each class name, docstring, and field description is sent to the model as part
# of the prompt, so they are written for the model to read. The validators run
# when LangChain parses the model's reply into the object; if one raises, the
# classify node catches it and the reflexion loop tries again.

class CpvClassification(BaseModel):
    """Split of the candidate CPV codes into the main code and the rest."""

    main_cpv: str = Field(
        description='The single main CPV code, copied verbatim from the candidate list, e.g. "71220000-6".',
    )
    additional_cpv: list[str] = Field(
        default_factory=list,
        description="Every other CPV code that applies to the procurement.",
    )
    reasoning: str = Field(
        description="One short sentence explaining the choice of main code.",
    )

    @field_validator("main_cpv")
    @classmethod
    def main_must_be_cpv(cls, value):
        """Reject a main code that is not in CPV form."""
        if not looks_like_cpv(value):
            raise ValueError(f"{value!r} is not a CPV code")
        return value

    @field_validator("additional_cpv")
    @classmethod
    def additional_must_be_cpv(cls, value):
        """Reject any additional code that is malformed."""
        bad = [code for code in value if not looks_like_cpv(code)]
        if bad:
            raise ValueError(f"not CPV codes: {bad}")
        return value

    @model_validator(mode="after")
    def main_not_in_additional(self):
        """Keep the main code out of the additional list."""
        if self.main_cpv:
            self.additional_cpv = [c for c in self.additional_cpv if c != self.main_cpv]
        return self


class Critique(BaseModel):
    """Self-assessment of a CPV classification (the reflexion step)."""

    verdict: Literal["accept", "revise"] = Field(
        description="accept if the split is well justified, otherwise revise."
    )
    problem: Optional[str] = Field(
        default=None,
        description="If revising, one short sentence on what to fix.",
    )


# MODEL
#
# Built once and reused. Constructing ChatOllama does not open a connection, so
# importing this module without Ollama running is fine.

chat_model = ChatOllama(model=EXTRACTION_MODEL, temperature=0, num_ctx=OLLAMA_NUM_CTX)
# method="json_schema" uses Ollama's constrained decoding, which fills the schema far
# more reliably on small local models than the default tool-calling path.
classifier = chat_model.with_structured_output(CpvClassification, method="json_schema")
critic = chat_model.with_structured_output(Critique, method="json_schema")


CLASSIFY_PROMPT = """You are reading CPV codes found in a Latvian public procurement \
notice. Each code is shown with the text around it.

CPV codes classify what is being procured. Exactly one code is the MAIN code (Latvian: \
"galvenais CPV kods", or simply "CPV kods").

You MUST choose exactly one code from the list as the main code and copy it verbatim. \
Put every other code in the additional list. Use only codes that appear in the list.

CODES:
{candidates}
{feedback}"""

CRITIQUE_PROMPT = """Check this CPV classification for a Latvian procurement notice.

CODES AND CONTEXT:
{candidates}

PROPOSED:
main CPV: {main}
additional CPV: {additional}
reasoning: {reasoning}

If the main code is the best-supported choice and the additional list is correct, answer \
accept. Otherwise answer revise and say briefly what to fix.
"""


# FILE READERS
#
# Each reader takes (bytes, filename) and returns text, or "" if it cannot read
# the file. READER_PIPELINE is the ordered list of attempts; OCR and .doc readers
# will be appended later without other changes.

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
                    yield from iter_container_files(payload, name, depth + 1)
                else:
                    yield name, payload
    except zipfile.BadZipFile:
        yield container_name, data


# GRAPH STATE
#
# One dict flows through every node. Nodes return only the keys they change, and
# LangGraph merges them in. No reducer is needed here: the reflexion loop replaces
# the classification each attempt rather than accumulating, so a plain overwrite
# is what we want.

class State(TypedDict):
    eis_id: str
    source_files: list[str]
    documents_text: str
    candidates: list[dict]          # [{code, context, count}]
    classification: Optional[dict]
    critique: Optional[dict]
    feedback: Optional[str]         # note carried into the next classify attempt
    attempts: int
    final: Optional[dict]


# HELPERS

def format_candidates(candidates):
    """Format the candidate codes and their context for a prompt."""
    lines = []
    for c in candidates:
        context = " ".join(c["context"].split())[:300]    # collapse whitespace, trim
        lines.append(f'- {c["code"]} (seen {c["count"]}x): {context}')
    return "\n".join(lines)


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


def find_candidates(state):
    """Find every CPV code in the text and keep the context around it."""
    text = state["documents_text"]
    found = {}
    for match in CPV_PATTERN.finditer(text):
        code = match.group()
        if code in found:
            found[code]["count"] += 1
            continue
        start = max(0, match.start() - CONTEXT_CHARS)
        end = min(len(text), match.end() + CONTEXT_CHARS)
        found[code] = {"code": code, "context": text[start:end], "count": 1}

    candidates = list(found.values())
    print(f"  find_candidates: {len(candidates)} unique CPV codes")
    return {"candidates": candidates}


def classify(state):
    """Ask the LLM to pick the main CPV code and list the additional ones."""
    attempt = state["attempts"] + 1
    feedback = state.get("feedback")
    note = f"\nNote on your previous attempt: {feedback}" if feedback else ""
    prompt = CLASSIFY_PROMPT.format(candidates=format_candidates(state["candidates"]), feedback=note)

    try:
        result = classifier.invoke(prompt)
    except ValidationError as error:
        print(f"  classify (attempt {attempt}): invalid output, will retry")
        return {"classification": None,
                "feedback": f"your previous answer was not valid: {error}",
                "attempts": attempt}

    # Keep only codes that were actually in the candidate list (no hallucinations).
    valid = {c["code"] for c in state["candidates"]}
    main = result.main_cpv if result.main_cpv in valid else None
    additional = [c for c in result.additional_cpv if c in valid]
    classification = {"main_cpv": main, "additional_cpv": additional, "reasoning": result.reasoning}

    print(f"  classify (attempt {attempt}): main {main}, additional {additional} (model said {result.main_cpv!r})")
    return {"classification": classification, "attempts": attempt}


def critique(state):
    """Judge the current classification and decide whether to revise it."""
    classification = state.get("classification")
    if not classification:
        # classify failed to produce valid output; ask for another attempt.
        return {"critique": {"verdict": "revise", "problem": "no valid classification produced"}}

    prompt = CRITIQUE_PROMPT.format(
        candidates=format_candidates(state["candidates"]),
        main=classification.get("main_cpv"),
        additional=classification.get("additional_cpv"),
        reasoning=classification.get("reasoning"),
    )
    result = critic.invoke(prompt)
    verdict = result.model_dump()

    print(f"  critique: {verdict['verdict']}" + (f" ({verdict['problem']})" if verdict.get("problem") else ""))
    update = {"critique": verdict}
    if verdict["verdict"] == "revise":
        update["feedback"] = verdict.get("problem") or "reconsider which code is the main CPV"
    return update


def route_after_critique(state):
    """Loop back to classify if the critique asked to revise and attempts remain."""
    if state["critique"]["verdict"] == "revise" and state["attempts"] < MAX_CLASSIFY_ATTEMPTS:
        return "classify"
    return "finalize"


def has_candidates(state):
    """Skip the LLM entirely when no CPV code was found."""
    return "classify" if state["candidates"] else "finalize"


def finalize(state):
    """Package the final result from the latest classification."""
    classification = state.get("classification")
    if not classification or not classification.get("main_cpv"):
        final = {"found": False, "main_cpv": None, "additional_cpv": [], "reasoning": None}
    else:
        final = {
            "found": True,
            "main_cpv": classification["main_cpv"],
            "additional_cpv": classification.get("additional_cpv", []),
            "reasoning": classification.get("reasoning"),
        }
    print(f"  finalize: main {final['main_cpv']}, additional {final['additional_cpv']}")
    return {"final": final}


# GRAPH ASSEMBLY

def build_graph():
    """Build and compile the extraction graph."""
    graph = StateGraph(State)
    graph.add_node("load_documents", load_documents)
    graph.add_node("find_candidates", find_candidates)
    graph.add_node("classify", classify)
    graph.add_node("critique", critique)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_documents")
    graph.add_edge("load_documents", "find_candidates")
    # Skip the LLM if there are no codes to classify.
    graph.add_conditional_edges(
        "find_candidates", has_candidates,
        {"classify": "classify", "finalize": "finalize"},
    )
    graph.add_edge("classify", "critique")
    # The reflexion loop: critique can send the answer back to classify.
    graph.add_conditional_edges(
        "critique", route_after_critique,
        {"classify": "classify", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

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
        "candidates": [],
        "classification": None,
        "critique": None,
        "feedback": None,
        "attempts": 0,
        "final": None,
    }


def process_one(graph, eis_id):
    """Run the graph for one procurement and save the result to disk."""
    print(f"Procurement {eis_id}")
    try:
        state = graph.invoke(initial_state(eis_id))
    except Exception as error:
        print(f"  pipeline failed: {error}")
        return

    result = {
        "eis_id": eis_id,
        "source_files": state.get("source_files", []),
        "candidates": state.get("candidates", []),   # kept for debugging
        "attempts": state.get("attempts", 0),
        "extracted": state.get("final"),
    }
    out_path = OUTPUT_DIR / f"{eis_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {out_path}\n")


def main():
    """Run extraction over the selected procurements."""
    parser = argparse.ArgumentParser(description="Extract CPV codes from procurements.")
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
# 1. Validation: a separate script reads extracted/{eis_id}.json and compares the
#    main code against the structured open data. additionalCpvType is often
#    missing from that data, so there is no clean answer key for it; the reflexion
#    step here is the quality mechanism in its place, and its real value should be
#    measured (does revising change the answer, or just add latency?).
#
# 2. More readers: append OCR (scanned PDFs) and a .doc reader to READER_PIPELINE.
#
# 3. Other fields: contract value, dates, and buyer could reuse this regex-then-
#    judge shape where a pattern exists, or fall back to the earlier chunked
#    LLM read for free text that has no reliable pattern.
#
# 4. Per-document classification feeding a conditional edge, the point where full
#    multi-agent orchestration enters the graph.