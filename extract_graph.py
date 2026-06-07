"""Extract the main and additional CPV codes from a procurement's documents.

This is a LangGraph pipeline. For one procurement it reads every document
(see readers.py), uses a regex to find every CPV code with its surrounding text,
then asks a local LLM (Ollama) which code is the main one and which are
additional. A reflection step lets the model critique its own answer and retry.

The reading layer also extracts any tables it finds. A separate node uses both the
located prose and those tables to extract the procurement's evaluation criteria.
The CPV task itself does not use the tables (CPV codes live in prose).

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
import re
from pathlib import Path
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from readers import iter_container_files, read_text, extract_tables


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")        # input: downloads/{eis_id}/*.zip
OUTPUT_DIR = Path("extracted")           # output: extracted/{eis_id}.json

EXTRACTION_MODEL = "mistral-small"       # any tag listed by `ollama list`
OLLAMA_NUM_CTX = 4096                    # Ollama's default of 2048 is too small

CONTEXT_CHARS = 200                      # characters of context kept on each side of a code
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

# Latvian (and English) cues for locating the criteria PROSE.
EVAL_KEYWORDS = [
    "vērtēšanas kritērij",
    "izvērtēšanas kritērij",
    "izvēles kritērij",
    "vērtēšanas metodik",
    "vērtēšanas kārtīb",
    "kritērija nosaukums",
    "punktu skait",
    "punktu piešķir",
    "saimnieciski visizdevīgāk",
    "zemākā cena",
    "zemāko piedāvāt",
    "evaluation criteri",
    "award criteri",
]

# Cues that mark a TABLE as a scoring table (matched against its cells).
CRITERIA_TABLE_CUES = ["kritērij", "punkt", "vērtēšan", "metodik"]

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
    award_method: Literal["lowest_price", "most_economically_advantageous", "unknown"] = Field(
        description="How the winning bid is chosen: lowest price, most economically advantageous, or unknown.",
    )
    criteria: list[Criterion] = Field(
        default_factory=list,
        description="The individual scoring criteria, if any.",
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

# A separate binding with a larger context window for the bigger criteria input.
criteria_model = ChatOllama(model=EXTRACTION_MODEL, temperature=0, num_ctx=CRITERIA_NUM_CTX)
criteria_extractor = criteria_model.with_structured_output(EvaluationCriteria, method="json_schema")


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

Extract the award method and the list of criteria with their weights. If no criteria are \
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
    source_files: list[str]
    documents_text: str
    tables: list[dict]              # tables found across all documents (not used for CPV)
    criteria: Optional[dict]        # extracted evaluation criteria
    criteria_attempts: int          # criteria reflexion loop counter
    criteria_feedback: Optional[str]
    criteria_check: Optional[str]   # "ok" or "revise" from the weight-sum check
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
    """Keep only scoring tables, de-duplicated across documents.

    A scoring table has several cells mentioning a criteria cue (kritērij, punkt,
    vērtēšan, metodik); this discards the many form and signature tables. The same
    table often appears in several files, so identical ones are dropped by a
    signature built from their short cells (criterion names, letters, weights),
    which is stable even when the long methodology text differs slightly.
    """
    chosen = []
    seen = set()
    for table in tables:
        cells = [c for row in table["rows"] for c in row]
        hits = sum(1 for c in cells if any(cue in c.lower() for cue in CRITERIA_TABLE_CUES))
        if hits < 2:
            continue
        signature = tuple(sorted(c.strip().lower() for c in cells if 0 < len(c.strip()) < 40))
        if signature in seen:
            continue
        seen.add(signature)
        chosen.append(table)
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
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def find_total_points(state, default=CRITERIA_TOTAL_DEFAULT):
    """Find the stated maximum total points, else fall back to the default.

    Targets the total line ("Maksimālais iespējamais punktu skaits: 100"); the
    word "iespējam" distinguishes it from the per-criterion maxima.
    """
    text = state.get("documents_text", "").lower()
    match = re.search(r"iespējam\w*\s+punktu\s+skait\w*\D{0,12}(\d{2,3})", text)
    return float(match.group(1)) if match else float(default)


# GRAPH NODES

def load_documents(state):
    """Read every leaf file into text, and collect any tables found."""
    folder = DOWNLOADS_DIR / state["eis_id"]
    texts = []
    file_names = []
    tables = []
    unread = []

    for zip_path in sorted(folder.glob("*.zip")):
        try:
            zip_bytes = zip_path.read_bytes()
        except OSError as error:
            print(f"    cannot read {zip_path.name}: {error}")
            continue

        for name, data in iter_container_files(zip_bytes, zip_path.name):
            text = read_text(data, name)
            if text:
                texts.append(f"Source file: {name}\n{text}")
                file_names.append(name)
            else:
                unread.append(name)
            tables.extend(extract_tables(data, name))

    blob = "\n\n".join(texts)
    message = (f"  load_documents: {len(blob):,} chars from {len(file_names)} files, "
               f"{len(tables)} tables")
    if unread:
        message += f" ({len(unread)} unreadable, e.g. {unread[0]!r})"
    print(message)
    return {"documents_text": blob, "source_files": file_names, "tables": tables}


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
    note = f"\nNote on your previous attempt: {feedback}" if feedback else ""

    if not prose and not tables:
        print("  extract_criteria: no criteria section or tables found")
        return {"criteria": {"found": False, "award_method": "unknown", "criteria": []},
                "criteria_attempts": attempt}

    prompt = CRITERIA_PROMPT.format(prose=prose or "(none found)",
                                    tables=tables or "(none found)", feedback=note)
    try:
        result = criteria_extractor.invoke(prompt)
        criteria = result.model_dump()
    except Exception as error:
        print(f"  extract_criteria: extraction failed: {error}")
        return {"criteria": {"found": False, "award_method": "unknown", "criteria": []},
                "criteria_attempts": attempt}

    print(f"  extract_criteria (attempt {attempt}): {len(criteria['criteria'])} criteria, "
          f"method {criteria['award_method']}")
    return {"criteria": criteria, "criteria_attempts": attempt}


def check_criteria(state):
    """Ground the criteria by checking their weights against the stated maximum.

    This is the external tool step that makes the criteria loop a reflexion agent
    rather than self-critique: a deterministic computation, not the model judging
    itself. If the weights do not reconcile, it asks for one more extraction.
    """
    criteria = state.get("criteria") or {}
    items = criteria.get("criteria") or []
    weights = [c["weight"] for c in items if c.get("weight") is not None]
    attempt = state.get("criteria_attempts", 0)

    if not weights:
        print("  check_criteria: no numeric weights to check")
        return {"criteria_check": "ok"}

    total = sum(weights)
    target = find_total_points(state)
    if abs(total - target) > 0.5 and attempt < MAX_CRITERIA_ATTEMPTS:
        print(f"  check_criteria: weights sum to {total:g}, expected {target:g}; revising")
        return {"criteria_check": "revise",
                "criteria_feedback": (f"the criteria weights sum to {total:g} but the maximum is "
                                      f"{target:g}; you may have missed a criterion or split one in two")}
    print(f"  check_criteria: weights sum to {total:g} (expected {target:g})")
    return {"criteria_check": "ok"}


def route_after_criteria_check(state):
    """Re-extract criteria if the weight check failed and an attempt remains."""
    return "extract_criteria" if state.get("criteria_check") == "revise" else "finalize"


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
    graph.add_node("extract_criteria", extract_criteria)
    graph.add_node("check_criteria", check_criteria)
    graph.add_node("find_candidates", find_candidates)
    graph.add_node("classify", classify)
    graph.add_node("critique", critique)
    # defer=True makes finalize wait for both parallel branches before running once
    # (otherwise it fires once per branch, since the branches finish at different times).
    graph.add_node("finalize", finalize, defer=True)

    graph.add_edge(START, "load_documents")

    # After loading, two independent branches run in parallel: evaluation criteria
    # and CPV codes. They write different state keys, so neither waits for the other.
    graph.add_edge("load_documents", "extract_criteria")
    graph.add_edge("load_documents", "find_candidates")

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

    # Both branches join here; finalize runs once after both complete.
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
        "tables": [],
        "criteria": None,
        "criteria_attempts": 0,
        "criteria_feedback": None,
        "criteria_check": None,
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
        "evaluation_criteria": state.get("criteria"),
        "tables": state.get("tables", []),            # captured for later structured-field work
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
# 1. Item-detail extraction: consume the captured tables with a reflexion agent
#    (extract line items, check them against the actual rows/totals, revise).
#    This is the natural home for true reflexion, since the table is external
#    ground truth to check against.
#
# 2. Per-document classification feeding a conditional edge (notice vs spec vs
#    financial proposal), to route documents to field-specific extractors. This
#    is where full multi-agent orchestration enters the graph.
#
# 3. Efficiency: a scanned PDF is currently rasterised twice (once for OCR text,
#    once for table detection). If that becomes a bottleneck, rasterise once and
#    share the page images between read_ocr and the table path.