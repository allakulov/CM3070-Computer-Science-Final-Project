"""Local procurement extraction: CPV reflection, lot-based criteria workers and standards.

The criteria branch uses fresh document-derived lots and saves detailed worker traces separately.
"""
import argparse
import os
import json
import math
import re
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from criteria_workers import run_criteria_workers
from lot_extraction import LotInventory, run_lot_extraction
from readers import CONTAINER_EXTS, IMAGE_EXTS, iter_container_files, read_file
from standards import find_standards
import sys

# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/ documents and archives
OUTPUT_DIR = Path("extracted")           # output: extracted/{eis_id}.json
TABLES_DIR = Path("tables")              # output: tables/{eis_id}.json    (captured tables, per file)
OCR_DIR = Path("ocr")                    # output: ocr/{eis_id}.json       (OCR text, per file)

EXTRACTION_MODEL = "gemma4:e4b"          # default model after evals
OLLAMA_NUM_CTX = 4096                    # context for CPV classification and critique

CONTEXT_CHARS = 200                      # characters of context kept on each side of a code
CANDIDATES_MAX_CHARS = 2048             # shared evidence budget, not a token count
FEEDBACK_MAX_CHARS = 400                # CPV retry feedback budget
CPV_REASONING_MAX_CHARS = 400
MAX_CLASSIFY_ATTEMPTS = 3                # cap on the reflection loop

# Criteria workers use the existing local model with a larger context.
CRITERIA_NUM_CTX = int(os.getenv("CRITERIA_NUM_CTX", "16384"))
CRITERIA_TRACE_DIR = Path(os.getenv("CRITERIA_TRACE_DIR", "criteria_worker_traces"))

class Tee:
    """Write everything to the terminal and to a log file at once."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# CPV code: 8 digits, a dash, then the check digit. the class also allows the
# unicode hyphen, non-breaking hyphen and en dash that OCR and PDF text sometimes
# produce instead of a plain hyphen.
CPV_REGEX = r"\d{8}[-\u2010\u2011\u2013]\d"
CPV_PATTERN = re.compile(r"(?<!\d)" + CPV_REGEX + r"(?!\d)")


def looks_like_cpv(code):
    """Return True if the string is exactly a CPV code."""
    return re.fullmatch(CPV_REGEX, code) is not None


# EXTRACTION SCHEMAS
#
# Each class name, docstring, and field description is sent to the model as part
# of the prompt, so they are written for the model to read. The validators run
# when LangChain parses the model's reply into the object; if one raises, the
# classify node catches it and the reflection loop tries again.

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
        """Keep a main code if in CPV form, return empty otherwise."""
        return value if looks_like_cpv(value) else ""

    @field_validator("additional_cpv")
    @classmethod
    def additional_must_be_cpv(cls, value):
        """Keep an additional code if in CPV form, drop otherwise."""
        return [code for code in value if looks_like_cpv(code)]

    @model_validator(mode="after")
    def main_not_in_additional(self):
        """Keep the main code out of the additional list."""
        if self.main_cpv:
            self.additional_cpv = [c for c in self.additional_cpv if c != self.main_cpv]
        return self


class Critique(BaseModel):
    """Self-assessment of a CPV classification (the reflection step)."""

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

classifier = critic = criteria_worker_model = lot_extractor = None   # set by build_models()


def build_models(name):
    """Build the CPV and criteria model bindings for one Ollama tag."""
    global classifier, critic, criteria_worker_model, lot_extractor
    # method="json_schema" uses Ollama's constrained decoding, which fills the schema
    # far more reliably on small local models than the default tool-calling path.
    chat_model = ChatOllama(model=name, temperature=0, num_ctx=OLLAMA_NUM_CTX)
    classifier = chat_model.with_structured_output(CpvClassification, method="json_schema")
    critic = chat_model.with_structured_output(Critique, method="json_schema")
    lot_model = ChatOllama(model=name, temperature=0, num_ctx=16384)
    lot_extractor = lot_model.with_structured_output(LotInventory, method="json_schema")
    # A larger context window for the bigger criteria input.
    criteria_model = ChatOllama(model=name, temperature=0, num_ctx=CRITERIA_NUM_CTX)
    criteria_worker_model = criteria_model


build_models(EXTRACTION_MODEL)


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

class State(TypedDict):
    eis_id: str
    downloads_dir: Path             # current procuremnt's folder 
    source_files: list[str]
    documents_text: str
    tables: list[dict]              # tables found across all documents, including CPV evidence
    ocr_records: list[dict]         # per-file OCR output, saved for evaluating the OCR model
    table_records: list[dict]       # per-file captured tables, saved for evaluating extraction
    standards: Optional[dict]       # standards and certificates found in the documents
    lots: Optional[dict]            # independently extracted lot inventory
    lot_seconds: float
    criteria: Optional[dict]        # extracted evaluation criteria
    criteria_check: Optional[str]   # numerical reconciliation, not factual acceptance
    candidates: list[dict]          # [{code, context, count}]
    classification: Optional[dict]
    cpv_error: Optional[dict]       # latest classification or critique failure
    critique: Optional[dict]
    feedback: Optional[str]         # note carried into the next classify attempt
    attempts: int
    cpv_seconds: float              # model time spent in the CPV classify/critique loop
    criteria_seconds: float         # model time spent extracting criteria
    final: Optional[dict]


# HELPERS

def format_candidates(candidates):
    """Keep all codes, then add whole windows while the shared budget permits.

    Prefer explicit main-code wording, independent of raw occurrence counts.
    None signals that even code-only lines cannot fit; callers must abstain.
    Character limits control growth but do not certify model token usage.
    """
    lines = [f'- {candidate["code"]}' for candidate in candidates]
    remaining = CANDIDATES_MAX_CHARS - len("\n".join(lines))
    if remaining < 0:
        return None
    # With no role cue, spread evidence across candidate positions instead of
    # always favouring the first codes. This is coverage, not prominence.
    spread = [0] if candidates else []
    unselected = set(range(1, len(candidates)))
    while unselected:
        index = max(sorted(unselected),
                    key=lambda i: min(abs(i - selected) for selected in spread))
        spread.append(index)
        unselected.remove(index)
    priority = {index: rank for rank, index in enumerate(spread)}
    evidence = []
    for index, candidate in enumerate(candidates):
        passages = candidate.get("contexts") or [candidate.get("context", "")]
        passages = list(dict.fromkeys(" ".join(p.split()) for p in passages if p.strip()))
        def is_main(passage):
            return bool(re.search(r"galven\w*\s+cpv|main\s+(?:cpv|code)", passage, re.I))
        # Prefer explicit role evidence even when it occurs between endpoints.
        ordered = sorted(enumerate(passages), key=lambda item: (
            not is_main(item[1]), item[0] not in (0, len(passages)-1), item[0]))
        for rank, (_, passage) in enumerate(ordered[:2]):
            evidence.append((not is_main(passage), rank, priority[index], index, passage))
    attached = set()
    for _, _, _, index, passage in sorted(evidence):
        separator = " || " if index in attached else ": "
        cost = len(separator) + len(passage)
        if cost <= remaining:
            lines[index] += separator + passage
            attached.add(index)
            remaining -= cost
    return "\n".join(lines)


def _cpv_budget_abstention(state, attempt):
    """Keep an oversized-input outcome visible in the existing saved record."""
    reason = (f"CPV classification not attempted: {len(state['candidates'])} candidate "
              f"codes exceed the {CANDIDATES_MAX_CHARS}-character code-list budget. "
              "Split this procurement by lot or document group and review it.")
    print(f"  classify: {reason}")
    return {"classification": {"main_cpv": None, "additional_cpv": [],
                                "reasoning": reason, "status": "input_budget_exceeded"},
            "attempts": attempt, "feedback": reason,
            "cpv_seconds": state.get("cpv_seconds", 0.0)}


# GRAPH NODES

def load_documents(state):
    """Read every leaf file into text, and collect any tables found."""
    folder = state["downloads_dir"] / state["eis_id"]
    texts = []
    file_names = []
    tables = []
    by_reader = {}       # reader label -> [names], what handled each read file
    ocr_files = []       # (name, char_count, preview) for the OCR'd files
    no_text = []         # (name, reason, table_count) for files that yielded no text
    ocr_records = []     # {name, chars, seconds, text} per OCR'd file, saved for evaluation
    table_records = []   # {name, seconds, tables} per file that produced tables

    supported = set(CONTAINER_EXTS + IMAGE_EXTS) | {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".txt", ".csv"}
    for path in sorted(folder.glob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            print(f"    cannot read {path.name}: {error}")
            continue
        leaves = (iter_container_files(payload, path.name)
                  if path.suffix.lower() in CONTAINER_EXTS
                  else [(path.name, payload)])
        for name, data in leaves:
            info = {}
            text, file_tables = read_file(data, name, info)
            if name.lower().endswith((".xlsx", ".xlsm")) and (text or file_tables):
                # Preserve worksheet scope without replacing the user's readers.py.
                # This adds one spreadsheet read; move into the reader at refactoring.
                try:
                    text, file_tables = spreadsheet_evidence(data, name)
                except Exception as error:
                    print(f"    worksheet context unavailable for {name}: {error}")
            if text or file_tables:
                file_names.append(name)
            if text:
                texts.append(f"Source file: {name}\n{text}")
                by_reader.setdefault(info["reader"], []).append(name)
                if info["reader"] == "ocr":
                    # A mixed PDF contains digital text too; save only its OCR here.
                    ocr_text = info.get("ocr_text", text)
                    preview = " ".join(ocr_text.split())[:120]
                    ocr_files.append((name, len(ocr_text), preview))
                    ocr_records.append({"name": name, "chars": len(ocr_text),
                                        "seconds": info["ocr_seconds"], "text": ocr_text})
            else:
                no_text.append((name, info.get("reason") or "unknown", len(file_tables)))
            if file_tables:
                table_records.append({"name": name, "seconds": info["parse_seconds"],
                                      "tables": [t["rows"] for t in file_tables]})
            tables.extend(file_tables)

    # Per-reader breakdown: which reader handled which files.
    for reader in ("pdf", "doc", "docx", "xls", "xlsx", "text"):
        names = by_reader.get(reader, [])
        if names:
            print(f"    {reader:6}({len(names)}): {', '.join(names)}")
    # OCR gets its own block with a character count and a short preview of the result.
    if ocr_files:
        print(f"    ocr   ({len(ocr_files)}):")
        for name, n_chars, preview in ocr_files:
            print(f"      {name}: {n_chars:,} chars | {preview!r}")
    # The full list of files that produced no text, with the reason where known.
    # A file can yield no prose but still contribute a table (e.g. a terse scoring
    # sheet), so that is noted rather than looking like the file was lost.
    if no_text:
        print(f"    no text({len(no_text)}):")
        for name, reason, n_tables in no_text:
            kept = f"  (+{n_tables} table{'s' if n_tables != 1 else ''} kept)" if n_tables else ""
            print(f"      {name}: {reason}{kept}")

    blob = "\n\n".join(texts)
    print(f"  load_documents: {len(blob):,} chars from {len(file_names)} files, "
          f"{len(tables)} tables")
    return {"documents_text": blob, "source_files": file_names, "tables": tables,
            "ocr_records": ocr_records, "table_records": table_records}

def extract_criteria(state):
    """Run one worker per observed lot using this run's reader output and inventory."""
    return run_criteria_workers(state, criteria_worker_model, CRITERIA_TRACE_DIR)


def extract_lots(state):
    return run_lot_extraction(state, lot_extractor)


def extract_standards(state):
    """Find the standards and certificates required in the documents.

    Deterministic (see standards.py), so this branch needs no model call and adds
    almost nothing to the runtime.
    """
    standards = find_standards(state["documents_text"], state["tables"])
    print(f"  extract_standards: {len(standards)} standards")
    return {"standards": standards}


def find_candidates(state):
    """Search prose and each table separately, keeping every occurrence window.

    Count is the number of matches across these representations; digital text
    and captured tables can overlap, so it is not a unique source count.
    """
    passages = [state.get("documents_text", "")]
    for table in state.get("tables", []):
        # Search cell content, not source filenames, for candidate codes.
        passages.append("\n".join(" | ".join(str(cell) for cell in row)
                                   for row in table.get("rows", [])))
    found = {}
    for text in passages:
        for match in CPV_PATTERN.finditer(text):
            code = match.group()
            start = max(0, match.start() - CONTEXT_CHARS)
            end = min(len(text), match.end() + CONTEXT_CHARS)
            context = text[start:end]
            candidate = found.setdefault(code, {"code": code, "context": context,
                                                "contexts": [], "count": 0})
            candidate["count"] += 1
            candidate["contexts"].append(context)
    candidates = list(found.values())
    print(f"  find_candidates: {len(candidates)} unique CPV codes")
    return {"candidates": candidates}


def classify(state):
    """Ask the LLM to pick the main CPV code and list the additional ones."""
    attempt = state["attempts"] + 1
    feedback = state.get("feedback")
    note = f"\nNote on your previous attempt: {str(feedback)[:FEEDBACK_MAX_CHARS]}" if feedback else ""
    started = time.perf_counter()
    try:
        candidate_text = format_candidates(state["candidates"])
        if candidate_text is None:
            return {**_cpv_budget_abstention(state, attempt), "cpv_error": None}
        prompt = CLASSIFY_PROMPT.format(candidates=candidate_text, feedback=note)
        result = classifier.invoke(prompt)
        valid = {c["code"] for c in state["candidates"]}
        main = result.main_cpv if result.main_cpv in valid else None
        additional = list(dict.fromkeys(c for c in result.additional_cpv if c in valid and c != main))
        classification = {"main_cpv": main, "additional_cpv": additional, "reasoning": result.reasoning}
    except Exception as error:
        detail = {"stage": "classify", "type": type(error).__name__, "message": str(error)}
        print(f"  classify (attempt {attempt}): error {type(error).__name__}: {error}")
        return {"classification": None, "critique": None, "cpv_error": detail,
                "feedback": f"Classification failed: {type(error).__name__}: {error}",
                "attempts": attempt,
                "cpv_seconds": round(state.get("cpv_seconds", 0.0) + time.perf_counter() - started, 1)}
    print(f"  classify (attempt {attempt}): main {main}, additional {additional} "
          f"(model said {result.main_cpv!r})")
    return {"classification": classification, "critique": None, "cpv_error": None,
            "feedback": None, "attempts": attempt,
            "cpv_seconds": round(state.get("cpv_seconds", 0.0) + time.perf_counter() - started, 1)}


def critique(state):
    """Judge the current classification and decide whether to revise it."""
    classification = state.get("classification")
    if not classification:
        # classify failed to produce valid output; ask for another attempt.
        return {"critique": {"verdict": "revise", "problem": "no valid classification produced"}}

    candidate_text = format_candidates(state["candidates"])
    if classification.get("status") == "input_budget_exceeded" or candidate_text is None:
        update = _cpv_budget_abstention(state, state.get("attempts", 0))
        update["critique"] = {"verdict": "revise", "problem": update["feedback"]}
        return update
    started = time.perf_counter()
    try:
        prompt = CRITIQUE_PROMPT.format(
            candidates=candidate_text, main=classification.get("main_cpv"),
            additional=classification.get("additional_cpv"),
            reasoning=str(classification.get("reasoning") or "")[:CPV_REASONING_MAX_CHARS])
        result = critic.invoke(prompt)
        verdict = result.model_dump()
    except Exception as error:
        detail = {"stage": "critique", "type": type(error).__name__, "message": str(error)}
        problem = f"Critique failed: {type(error).__name__}: {error}"
        print(f"  critique: {problem}")
        return {"critique": {"verdict": "revise", "problem": problem}, "cpv_error": detail,
                "feedback": problem,
                "cpv_seconds": round(state.get("cpv_seconds", 0.0) + time.perf_counter() - started, 1)}

    print(f"  critique: {verdict['verdict']}" + (f" ({verdict['problem']})" if verdict.get("problem") else ""))
    update = {"critique": verdict, "cpv_error": None, "feedback": None,
              "cpv_seconds": round(state.get("cpv_seconds", 0.0) + time.perf_counter() - started, 1)}
    if verdict["verdict"] == "revise":
        update["feedback"] = verdict.get("problem") or "reconsider which code is the main CPV"
    return update


def route_after_critique(state):
    """Loop back to classify if the critique asked to revise and attempts remain."""
    if (state.get("classification") or {}).get("status") == "input_budget_exceeded":
        return "finalize"
    if state["critique"]["verdict"] == "revise" and state["attempts"] < MAX_CLASSIFY_ATTEMPTS:
        return "classify"
    return "finalize"


def has_candidates(state):
    """Skip the LLM entirely when no CPV code was found."""
    return "classify" if state["candidates"] else "finalize"


def finalize(state):
    """Keep recovered codes and report whether the CPV branch accepted them.

    found indicates a main code is present, not that review succeeded. Consumers
    must use status to distinguish accepted, partial, unresolved and error results.
    """
    classification = state.get("classification") or {}
    main = classification.get("main_cpv") or None
    additional = list(dict.fromkeys(c for c in classification.get("additional_cpv", []) if c != main))
    critique = state.get("critique") or {}
    error = state.get("cpv_error")
    if classification.get("status") == "input_budget_exceeded":
        status = "input_budget_exceeded"
    elif error:
        status = "error"
    elif critique.get("verdict") == "revise":
        status = "unresolved"
    elif not main and additional:
        status = "partial"
    elif main and critique.get("verdict") == "accept":
        status = "accepted"
    elif classification:
        status = "not_reviewed"
    else:
        status = "no_candidates" if not state.get("candidates") else "unresolved"
    final = {"found": bool(main), "main_cpv": main, "additional_cpv": additional,
             "reasoning": classification.get("reasoning"), "status": status,
             "error": error,
             "review_problem": critique.get("problem"),
             "retry_exhausted": status in {"error", "unresolved"}
                 and state.get("attempts", 0) >= MAX_CLASSIFY_ATTEMPTS}
    print(f"  finalize: status={status}, main {main}, additional {additional}")
    return {"final": final}


# GRAPH ASSEMBLY

def build_graph():
    """Build and compile the extraction graph."""
    graph = StateGraph(State)
    graph.add_node("load_documents", load_documents)
    graph.add_node("extract_lots", extract_lots)
    graph.add_node("extract_criteria", extract_criteria)
    graph.add_node("find_candidates", find_candidates)
    graph.add_node("classify", classify)
    graph.add_node("critique", critique)
    graph.add_node("extract_standards", extract_standards)
    # defer=True makes finalize wait for every parallel branch before running once
    # (otherwise it fires once per branch, since the branches finish at different times).
    graph.add_node("finalize", finalize, defer=True)

    graph.add_edge(START, "load_documents")

    # After loading, three independent branches run in parallel: evaluation criteria,
    # CPV codes, and standards. They write different state keys, so none waits for
    # the others.
    graph.add_edge("load_documents", "extract_lots")
    graph.add_edge("extract_lots", "extract_criteria")
    graph.add_edge("load_documents", "find_candidates")
    graph.add_edge("load_documents", "extract_standards")

    # Criteria workers finish once; their collector reports unresolved lots.
    graph.add_edge("extract_criteria", "finalize")

    # CPV branch: its reflection loop. Skip the LLM if there are no codes.
    graph.add_conditional_edges(
        "find_candidates", has_candidates,
        {"classify": "classify", "finalize": "finalize"},
    )
    graph.add_edge("classify", "critique")
    graph.add_conditional_edges(
        "critique", route_after_critique,
        {"classify": "classify", "finalize": "finalize"},
    )

    # Standards branch: deterministic, so it runs straight through with no loop.
    graph.add_edge("extract_standards", "finalize")

    # All three branches join here; finalize runs once after they complete.
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

def initial_state(eis_id, downloads_dir=DOWNLOADS_DIR):
    """Return a fresh state for one procurement."""
    return {
        "eis_id": eis_id,
        "downloads_dir": downloads_dir,
        "source_files": [],
        "documents_text": "",
        "tables": [],
        "ocr_records": [],
        "table_records": [],
        "lots": None,
        "lot_seconds": 0.0,
        "criteria": None,
        "criteria_check": None,
        "candidates": [],
        "classification": None,
        "cpv_error": None,
        "critique": None,
        "feedback": None,
        "attempts": 0,
        "cpv_seconds": 0.0,
        "criteria_seconds": 0.0,
        "final": None,
         "standards": None,
    }


def build_record(eis_id, state):
    """Build the extraction record for one procurement from its finished run state. The CLI and the
    interface both call this, prevents difference between them.
    """
    return {
        "eis_id": eis_id,
        "source_files": state.get("source_files", []),
        "candidates": state.get("candidates", []),   # kept for debugging
        "attempts": state.get("attempts", 0),
        "model_seconds": round(state.get("cpv_seconds", 0.0) + state.get("criteria_seconds", 0.0) + state.get("lot_seconds", 0.0), 1),
        "extracted": state.get("final"),
        "critique": state.get("critique"),
        "lots": state.get("lots"),
        "lot_seconds": state.get("lot_seconds", 0.0),
        "evaluation_criteria": state.get("criteria"),
        "criteria_check": state.get("criteria_check"),
        "standards": state.get("standards"),
        }


def save_result(eis_id, state, out_dir=OUTPUT_DIR, tables_dir=TABLES_DIR, ocr_dir=OCR_DIR):
    """Write the record and its captured tables and OCR to the three output folders.
    The extraction files are the same for the CLI and the interface both call this.
    """
    for folder in (out_dir, tables_dir, ocr_dir):
        folder.mkdir(parents=True, exist_ok=True)
    record = build_record(eis_id, state)
    out_path = out_dir / f"{eis_id}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Raw captured material is saved apart from the agents' conclusions above, so the
    # OCR and table-extraction models each have their own artifact to evaluate against.
    tables_path = tables_dir / f"{eis_id}.json"
    tables_path.write_text(json.dumps(state.get("table_records", []), indent=2,
                                      ensure_ascii=False), encoding="utf-8")
    ocr_path = ocr_dir / f"{eis_id}.json"
    ocr_path.write_text(json.dumps(state.get("ocr_records", []), indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    return out_path, tables_path, ocr_path


def process_one(graph, eis_id):
    """Run the graph for one procurement and save the result to disk."""
    print(f"Procurement {eis_id}")
    try:
        state = graph.invoke(initial_state(eis_id))
    except Exception as error:
        print(f"  pipeline failed: {error}")
        return

    out_path, tables_path, ocr_path = save_result(eis_id, state, OUTPUT_DIR, TABLES_DIR, OCR_DIR)
    print(f"  saved {out_path}, {tables_path}, {ocr_path}\n")


def main():
    """Run extraction over the selected procurements."""
    global OUTPUT_DIR, TABLES_DIR, OCR_DIR, CRITERIA_TRACE_DIR
    parser = argparse.ArgumentParser(description="Extract structured fields from procurements.")
    parser.add_argument("--id", nargs="+", help="One or more procurement ids to process")
    parser.add_argument("--ids-file", help="A file with one procurement id per line")
    parser.add_argument("--model", default=EXTRACTION_MODEL, help="Ollama model tag to extract with")
    parser.add_argument("--output-root", help="Save this run in a separate directory")
    args = parser.parse_args()
    run_root = Path(args.output_root or ".")
    run_root.mkdir(parents=True, exist_ok=True)
    if args.output_root:
        OUTPUT_DIR = run_root / "extracted"
        TABLES_DIR = run_root / "tables"
        OCR_DIR = run_root / "ocr"
        CRITERIA_TRACE_DIR = run_root / "criteria_worker_traces"
    sys.stdout = Tee(run_root / "extraction_terminal_logs.txt")

    if args.model != EXTRACTION_MODEL:
        build_models(args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)     # extracted/, not a per-model subfolder
    TABLES_DIR.mkdir(exist_ok=True)
    OCR_DIR.mkdir(exist_ok=True)

    eis_ids = list(args.id or [])
    if args.ids_file:
        ids_path = Path(args.ids_file)
        if not ids_path.is_file():
            print(f"ids file not found: {ids_path}")
            return
        eis_ids += [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if not eis_ids:
        eis_ids = sorted(p.name for p in DOWNLOADS_DIR.iterdir() if p.is_dir())
    eis_ids = list(dict.fromkeys(eis_ids))            # drop duplicates, keep order

    if not eis_ids:
        print(f"No procurement found in {DOWNLOADS_DIR}")
        return

    graph = build_graph()
    save_graph_image(graph, run_root / "graph.png")
    print()

    (run_root / "requested_ids.txt").write_text("\n".join(eis_ids) + "\n", encoding="utf-8")
    for eis_id in eis_ids:
        if (DOWNLOADS_DIR / eis_id).is_dir():
            process_one(graph, eis_id)
        else:
            print(f"skipping {eis_id}: no folder in {DOWNLOADS_DIR}")


if __name__ == "__main__":
    main()
