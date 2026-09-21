"""Extract the main and additional CPV codes from a procurement's documents.

This is a LangGraph pipeline. For one procurement it reads every document
(see readers.py), uses a regex to find every CPV code with its surrounding text,
then asks a local LLM (Ollama) which code is the main one and which are
additional. A reflection step lets the model critique its own answer and retry.

The reading layer also extracts any tables it finds. A separate node uses both the
located prose and those tables to extract the procurement's evaluation criteria.
The CPV task searches prose and captured tables, retaining repeated evidence.

Why a regex first, then an LLM: the regex catches every code and gives the
evidence text for free, so the LLM only judges a short candidate list rather
than scanning the whole document. The model never sees the full text, so no
chunking is needed for this field.

Install:
    pip install langgraph langchain-ollama pydantic
    pip install pdfplumber python-docx openpyxl pymupdf    # base readers
    pip install easyocr                                     # OCR fallback (scanned PDFs)
    pip install transformers torch torchvision              # optional: Table Transformer
    ollama pull mistral-small

Run:
    python extract_graph.py                  # every procurement in downloads/
    python extract_graph.py --eis-id 123450  # just one
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from readers import iter_container_files, read_file
from standards import find_standards
import sys

# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/ documents and archives
OUTPUT_DIR = Path("extracted")           # output: extracted/{model}/{eis_id}.json (per model)
TABLES_DIR = Path("tables")              # output: tables/{eis_id}.json    (captured tables, per file)
OCR_DIR = Path("ocr")                    # output: ocr/{eis_id}.json       (OCR text, per file)

EXTRACTION_MODEL = "gemma4:e4b"          # default model after evals
OLLAMA_NUM_CTX = 4096                    # Ollama's default of 2048 is too small

CONTEXT_CHARS = 200                      # characters of context kept on each side of a code
CANDIDATES_MAX_CHARS = 2048             # shared evidence budget, not a token count
FEEDBACK_MAX_CHARS = 400                # both extraction retry loops
CPV_REASONING_MAX_CHARS = 400
MAX_CLASSIFY_ATTEMPTS = 3                # cap on the reflection loop

# Evaluation-criteria extraction. Its input (a prose section plus tables) is larger
# than CPV's tiny candidate list, so it gets a bigger context window. We locate the
# criteria section by keyword rather than feeding the whole document.
CRITERIA_NUM_CTX = 8192
CRITERIA_WINDOW = 2000                   # chars of prose kept around each keyword hit
CRITERIA_PROSE_MAX = 6000                # cap on total prose fed to the criteria node
CRITERIA_TABLES_MAX = 6000               # cap on total table markdown fed to the criteria node
CRITERIA_CELL_CHARS = 200                # truncate each table cell (scoring methodology is huge)
MAX_CRITERIA_ATTEMPTS = 2                # cap on the criteria reflexion loop
CRITERIA_TOTAL_DEFAULT = 100             # assumed maximum points if the notice does not state one
CRITERIA_DEDUP_OVERLAP = 0.6             # drop a scoring table this much covered by a fuller one

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

# Cues for locating the criteria PROSE, kept to the terms that actually fired across
# the corpus (see inspect_criteria.py). "vērtēšanas kritērij" also matches inside
# "izvērtēšanas kritērij"; "visizdevīgāk" covers the most-economically-advantageous
# phrasings and "viszemāk" the lowest-price ones.
EVAL_KEYWORDS = [
    "vērtēšanas kritērij",
    "izvēles kritērij",
    "visizdevīgāk",
    "viszemāk",
    "punktu skait",
]

# Cues that mark a TABLE as a scoring table (matched against its cells).
CRITERIA_TABLE_CUES = ["kritērij", "punkt", "vērtēšan", "metodik"]

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

    # @field_validator("main_cpv")
    # @classmethod
    # def main_must_be_cpv(cls, value):
    #     """Reject a main code that is not in CPV form."""
    #     if not looks_like_cpv(value):
    #         raise ValueError(f"{value!r} is not a CPV code")
    #     return value
    @field_validator("main_cpv")
    @classmethod
    def main_must_be_cpv(cls, value):
        """Keep a main code if in CPV form, return empty otherwise."""
        return value if looks_like_cpv(value) else ""

    # @field_validator("additional_cpv")
    # @classmethod
    # def additional_must_be_cpv(cls, value):
    #     """Reject any additional code that is malformed."""
    #     bad = [code for code in value if not looks_like_cpv(code)]
    #     if bad:
    #         raise ValueError(f"not CPV codes: {bad}")
    #     return value
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


class Criterion(BaseModel):
    """One evaluation criterion used to score bids."""

    name: str = Field(description="The criterion, e.g. price, quality, delivery time.")
    weight: Optional[float] = Field(
        default=None,
        description="Numeric weight in points or percent, if stated; otherwise null.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Any short detail or formula, if given.",
    )


class EvaluationCriteria(BaseModel):
    """How bids are scored in a procurement notice."""

    found: bool = Field(description="True if evaluation criteria are present in the text.")
    criteria: list[Criterion] = Field(
        default_factory=list,
        description="The individual scoring criteria, if any.",
    )


# MODEL
#
# Built once and reused. Constructing ChatOllama does not open a connection, so
# importing this module without Ollama running is fine.

classifier = critic = criteria_extractor = None   # set by build_models()


def build_models(name):
    """Build the CPV and criteria model bindings for one Ollama tag."""
    global classifier, critic, criteria_extractor
    # method="json_schema" uses Ollama's constrained decoding, which fills the schema
    # far more reliably on small local models than the default tool-calling path.
    chat_model = ChatOllama(model=name, temperature=0, num_ctx=OLLAMA_NUM_CTX)
    classifier = chat_model.with_structured_output(CpvClassification, method="json_schema")
    critic = chat_model.with_structured_output(Critique, method="json_schema")
    # A larger context window for the bigger criteria input.
    criteria_model = ChatOllama(model=name, temperature=0, num_ctx=CRITERIA_NUM_CTX)
    criteria_extractor = criteria_model.with_structured_output(EvaluationCriteria, method="json_schema")


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

CRITERIA_PROMPT = """You are reading a Latvian public procurement notice to find its \
EVALUATION CRITERIA -- the rules used to score bids. These may appear in the prose, in a \
table, or both.

Latvian cues: "vērtēšanas kritēriji", "piedāvājuma izvēles kritērijs", "saimnieciski \
visizdevīgākais piedāvājums" (most economically advantageous), "zemākā cena" (lowest \
price). Criteria often carry weights in percent or points.

Extract the list of criteria with their weights. If no criteria are \
present, set found to false.

PROSE:
{prose}

TABLES:
{tables}
{feedback}"""


# GRAPH STATE
#
# One dict flows through every node. Nodes return only the keys they change, and
# LangGraph merges them in. No reducer is needed here: the reflection loop replaces
# the classification each attempt rather than accumulating, so a plain overwrite
# is what we want.

class State(TypedDict):
    eis_id: str
    downloads_dir: Path             # current procuremnt's folder 
    source_files: list[str]
    documents_text: str
    tables: list[dict]              # tables found across all documents, including CPV evidence
    ocr_records: list[dict]         # per-file OCR output, saved for evaluating the OCR model
    table_records: list[dict]       # per-file captured tables, saved for evaluating extraction
    standards: Optional[dict]       # standards and certificates found in the documents
    criteria: Optional[dict]        # extracted evaluation criteria
    criteria_attempts: int          # criteria reflexion loop counter
    criteria_feedback: Optional[str]
    criteria_check: Optional[str]   # "ok" or "revise" from the weight-sum check
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


def locate_sections(text, keywords, window=CRITERIA_WINDOW, max_total=CRITERIA_PROSE_MAX):
    """Return windows of text around keyword hits, merged and capped.

    The same locate-then-focus idea as CPV: instead of feeding the whole document,
    keep only the parts near a keyword so the relevant section reaches the model.
    """
    low = text.lower()
    spans = []
    for keyword in keywords:
        start = 0
        while True:
            hit = low.find(keyword, start)
            if hit == -1:
                break
            spans.append((max(0, hit - window // 2), min(len(text), hit + window // 2)))
            start = hit + len(keyword)
    if not spans:
        return ""

    spans.sort()
    merged = [spans[0]]
    for begin, end in spans[1:]:
        if begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))

    pieces = []
    total = 0
    for begin, end in merged:
        piece = text[begin:end]
        if total + len(piece) > max_total:
            piece = piece[: max_total - total]
        pieces.append(piece)
        total += len(piece)
        if total >= max_total:
            break
    return "\n...\n".join(pieces)


def _cell(value, limit=CRITERIA_CELL_CHARS):
    """Make one cell safe and short for a markdown table row.

    Cells are truncated because a scoring table's methodology cells run to
    hundreds of characters; the criterion name and weight sit at the start, so a
    short prefix keeps what matters and leaves budget for every criterion.
    """
    text = str(value).replace("|", "/").replace("\n", " ").strip()
    return text[:limit]


def select_criteria_tables(tables):
    """Keep distinct scoring tables, dropping copies and fragments of the same one.

    A scoring table has at least two cells hitting a criteria cue (kritērij, punkt,
    vērtēšan, metodik); this discards the form and signature tables. The catch found
    on real notices: the one real criteria table is repeated across files (the
    regulations, the report, the EIS export) and fragmented differently by page
    breaks, so an exact-signature match misses the copies. Instead we compare each
    table's short cells (criterion names, letters, weights) as a set: considering the
    fullest tables first, a later table whose short cells are mostly already covered
    by a kept one is a copy or fragment and is dropped, while a genuinely different
    (complementary) table is kept.
    """
    # Collect each scoring table with the set of its short, identifying cells.
    scoring = []
    for table in tables:
        cells = [c for row in table["rows"] for c in row]
        hits = sum(1 for c in cells if any(cue in c.lower() for cue in CRITERIA_TABLE_CUES))
        if hits < 2:
            continue                                     # not a scoring table
        short = {c.strip().lower() for c in cells if 0 < len(c.strip()) < 40}
        if short:
            scoring.append((short, table))

    # Fullest first, so fragments and copies are measured against the richer table.
    scoring.sort(key=lambda pair: len(pair[0]), reverse=True)
    chosen, chosen_sets = [], []
    for short, table in scoring:
        covered = max((len(short & kept) / len(short) for kept in chosen_sets), default=0)
        if covered < CRITERIA_DEDUP_OVERLAP:
            chosen.append(table)
            chosen_sets.append(short)
    return chosen


def tables_to_markdown(tables, max_chars=CRITERIA_TABLES_MAX):
    """Render tables as markdown (header preserved), capped to a character budget."""
    blocks = []
    total = 0
    for table in tables:
        rows = table["rows"]
        if not rows:
            continue
        header = rows[0]
        lines = [f"Table from {table['source']}:",
                 "| " + " | ".join(_cell(c) for c in header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for row in rows[1:]:
            lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
        block = "\n".join(lines)
        separator = 2 if blocks else 0
        if total + separator + len(block) > max_chars:
            continue            # a later, smaller table may still fit
        blocks.append(block)
        total += separator + len(block)
    return "\n\n".join(blocks)


def find_total_points(state, default=CRITERIA_TOTAL_DEFAULT):
    """Find the first explicit total in prose, then table rows; preserve decimals.

    The Latvian total cue distinguishes this from individual criterion maxima.
    Permit short same-line wording between the cue and value. Positive totals
    may have any digit length; conflicting/lot-specific totals need a policy.
    """
    passages = [state.get("documents_text", "")]
    passages.extend(" | ".join(str(cell or "") for cell in row)
                    for table in state.get("tables", []) for row in table.get("rows", []))
    pattern = r"iespējam\w*[^\S\r\n]+punktu[^\S\r\n]+skait\w*([^\d\r\n]{0,20}?)(\d+(?:[.,]\d+)?)(?!\d|[.,]\d)"
    for passage in passages:
        for match in re.finditer(pattern, passage, re.I):
            # A spaced dash is punctuation; an adjacent minus is a sign.
            # Do not turn "-60" into a positive total while accepting "- 60".
            if match.group(1).endswith(("-", "−", "–")):
                continue
            value = float(match.group(2).replace(",", "."))
            if math.isfinite(value) and value > 0:
                return value
    return float(default)


# GRAPH NODES

def load_documents(state):
    """Read every leaf file into text, and collect any tables found."""
    # folder = DOWNLOADS_DIR / state["eis_id"]
    folder = state["downloads_dir"] / state["eis_id"]
    texts = []
    file_names = []
    tables = []
    by_reader = {}       # reader label -> [names], what handled each read file
    ocr_files = []       # (name, char_count, preview) for the OCR'd files
    no_text = []         # (name, reason, table_count) for files that yielded no text
    ocr_records = []     # {name, chars, seconds, text} per OCR'd file, saved for evaluation
    table_records = []   # {name, seconds, tables} per file that produced tables

    supported = {".zip", ".edoc", ".pdf", ".doc", ".docx", ".xlsx", ".xlsm",
                 ".txt", ".csv", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    for path in sorted(folder.glob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            print(f"    cannot read {path.name}: {error}")
            continue
        leaves = (iter_container_files(payload, path.name)
                  if path.suffix.lower() in {".zip", ".edoc"}
                  else [(path.name, payload)])
        for name, data in leaves:
            info = {}
            text, file_tables = read_file(data, name, info)    # one parse per file
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
    for reader in ("pdf", "doc", "docx", "xlsx", "text"):
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

def extract_standards(state):
    """Find the standards and certificates required in the documents.

    Deterministic (see standards.py), so this branch needs no model call and adds
    almost nothing to the runtime.
    """
    standards = find_standards(state["documents_text"], state["tables"])
    print(f"  extract_standards: {len(standards)} standards")
    return {"standards": standards}

def extract_criteria(state):
    """Extract the evaluation criteria from the located prose and the scoring tables.

    Criteria can live in prose, a table, or both, so the node sees both: a window
    of prose around the criteria keywords plus the de-duplicated scoring tables as
    markdown. On a re-run, the previous attempt's weight mismatch is fed back.
    """
    prose = locate_sections(state["documents_text"], EVAL_KEYWORDS)
    tables = tables_to_markdown(select_criteria_tables(state["tables"]))
    attempt = state.get("criteria_attempts", 0) + 1
    feedback = state.get("criteria_feedback")
    note = f"\nNote on your previous attempt: {str(feedback)[:FEEDBACK_MAX_CHARS]}" if feedback else ""

    if not prose and not tables:
        print("  extract_criteria: no criteria section or tables found")
        return {"criteria": {"found": False, "criteria": [], "extraction_status": "no_input"},
                "criteria_attempts": attempt}

    prompt = CRITERIA_PROMPT.format(prose=prose or "(none found)",
                                    tables=tables or "(none found)", feedback=note)
    started = time.perf_counter()
    try:
        result = criteria_extractor.invoke(prompt)
        criteria = result.model_dump()
    except Exception as error:
        secs = round(state.get("criteria_seconds", 0.0) + time.perf_counter() - started, 1)
        print(f"  extract_criteria: extraction failed: {error}")
        return {"criteria": {"found": False, "criteria": [], "extraction_status": "error",
                             "error": f"{type(error).__name__}: {error}"},
                "criteria_attempts": attempt, "criteria_seconds": secs}
    secs = round(state.get("criteria_seconds", 0.0) + time.perf_counter() - started, 1)

    criteria["extraction_status"] = "completed"
    print(f"  extract_criteria (attempt {attempt}): {len(criteria['criteria'])} criteria")
    return {"criteria": criteria, "criteria_attempts": attempt, "criteria_seconds": secs}


def _normalize_name(name):
    """Normalize case and punctuation without discarding role qualifiers.

    Parentheses may contain roles, not translations. Translation equivalence
    needs explicit evidence and is not inferred by deleting text.
    """
    return " ".join(re.sub(r"[^\w ]", " ", name.lower()).split())


def dedup_criteria(items):
    """Drop criteria that share a normalised name, keeping the first seen."""
    seen = {}
    for c in items:
        key = _normalize_name(c.get("name", ""))
        if key and key not in seen:
            seen[key] = c
    return list(seen.values())


def check_criteria(state):
    """Separate routing from validation, preserving explicit weights and errors.

    A sole criterion with no stated weight defaults to 100. Explicit weights
    are preserved; earlier versions overwrote any sole weight.
    Reconciliation checks numeric completeness and totals, not factual accuracy.
    """
    original = state.get("criteria") or {}
    items = [dict(item) for item in dedup_criteria(original.get("criteria") or [])]
    criteria = {**original, "criteria": items, "weights_reconcile": None,
                "weights_total": None, "expected_total": find_total_points(state)}
    attempt = state.get("criteria_attempts", 0)
    if criteria.get("extraction_status") == "error":
        criteria["validation_status"] = "extraction_error"
        return {"criteria": criteria, "criteria_check": "ok", "criteria_feedback": None}
    if not items:
        criteria["validation_status"] = "no_criteria"
        return {"criteria": criteria, "criteria_check": "ok", "criteria_feedback": None}
    if len(items) == 1 and items[0].get("weight") is None:
        items[0]["weight"] = 100
        items[0]["weight_origin"] = "sole_criterion_default"
    supplied = [item.get("weight") for item in items]
    missing = sum(weight is None for weight in supplied)
    invalid = any(weight is not None and (
        isinstance(weight, bool) or not isinstance(weight, (int, float))
        or not math.isfinite(weight) or weight < 0) for weight in supplied)
    weights = [weight for weight in supplied if weight is not None]
    total = sum(weights) if weights and not invalid else None
    target = criteria["expected_total"]
    criteria["weights_total"] = total
    if invalid:
        status = "invalid_weights"
    elif missing:
        status = "missing_weights"
    elif abs(total - target) > 0.5:
        status = "total_mismatch"
    else:
        status = "reconciled"
    criteria["validation_status"] = status
    criteria["weights_reconcile"] = status == "reconciled"
    # Keep the existing stop policy when every weight is missing. Do not ask
    # the model to invent weights merely to satisfy the expected total.
    retry = status != "reconciled" and bool(weights) and attempt < MAX_CRITERIA_ATTEMPTS
    criteria["retry_exhausted"] = status != "reconciled" and bool(weights) and not retry
    total_text = f"{total:g}" if total is not None else "unknown"
    feedback = (f"Check {status}: sum {total_text}, target {target:g}, missing {missing}/{len(items)}. "
                "Merge entries only if the source confirms the same criterion, not distinct roles. "
                f"Verify each weight is its share of {target:g}, not a per-criterion maximum. "
                "Preserve stated weights; leave unstated values null. "
                "Do not invent weights to force the sum.") if retry else None

    print(f"  check_criteria: {status}, total {total}, expected {target:g}")
    return {"criteria": criteria, "criteria_check": "revise" if retry else "ok",
            "criteria_feedback": feedback}


def route_after_criteria_check(state):
    """Re-extract criteria if the weight check failed and an attempt remains."""
    return "extract_criteria" if state.get("criteria_check") == "revise" else "finalize"


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
    graph.add_node("extract_criteria", extract_criteria)
    graph.add_node("check_criteria", check_criteria)
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
    graph.add_edge("load_documents", "extract_criteria")
    graph.add_edge("load_documents", "find_candidates")
    graph.add_edge("load_documents", "extract_standards")

    # Criteria branch: a reflexion loop grounded by the weight-sum check.
    graph.add_edge("extract_criteria", "check_criteria")
    graph.add_conditional_edges(
        "check_criteria", route_after_criteria_check,
        {"extract_criteria": "extract_criteria", "finalize": "finalize"},
    )

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
        "criteria": None,
        "criteria_attempts": 0,
        "criteria_feedback": None,
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
        "model_seconds": round(state.get("cpv_seconds", 0.0) + state.get("criteria_seconds", 0.0), 1),
        "extracted": state.get("final"),
        "critique": state.get("critique"),
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

    # result = {
    #     "eis_id": eis_id,
    #     "source_files": state.get("source_files", []),
    #     "candidates": state.get("candidates", []),   # kept for debugging
    #     "attempts": state.get("attempts", 0),
    #     "model_seconds": round(state.get("cpv_seconds", 0.0) + state.get("criteria_seconds", 0.0), 1),
    #     "extracted": state.get("final"),
    #     "critique": state.get("critique"),
    #     "evaluation_criteria": state.get("criteria"),
    #     "criteria_check": state.get("criteria_check"),
    #     "standards": state.get("standards"),
    # }
    # out_path = OUTPUT_DIR / f"{eis_id}.json"
    # out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # # Raw captured material is saved apart from the agents' conclusions above, so the
    # # OCR and table-extraction models each have their own artifact to evaluate against.
    # tables_path = TABLES_DIR / f"{eis_id}.json"
    # tables_path.write_text(json.dumps(state.get("table_records", []), indent=2,
    #                                   ensure_ascii=False), encoding="utf-8")
    # ocr_path = OCR_DIR / f"{eis_id}.json"
    # ocr_path.write_text(json.dumps(state.get("ocr_records", []), indent=2,
    #                                ensure_ascii=False), encoding="utf-8")
    out_path, tables_path, ocr_path = save_result(eis_id, state)
    print(f"  saved {out_path}, {tables_path}, {ocr_path}\n")


def main():
    """Run extraction over the selected procurements."""
    sys.stdout = Tee("extraction_terminal_logs.txt")
    parser = argparse.ArgumentParser(description="Extract structured fields from procurements.")
    parser.add_argument("--id", nargs="+", help="One or more procurement ids to process")
    parser.add_argument("--ids-file", help="A file with one procurement id per line")
    parser.add_argument("--model", default=EXTRACTION_MODEL, help="Ollama model tag to extract with")
    args = parser.parse_args()

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
    save_graph_image(graph)
    print()

    for eis_id in eis_ids:
        if (DOWNLOADS_DIR / eis_id).is_dir():
            process_one(graph, eis_id)
        else:
            print(f"skipping {eis_id}: no folder in {DOWNLOADS_DIR}")


if __name__ == "__main__":
    main()
