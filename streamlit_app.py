"""Streamlit UI

This app provides a UI for extract_graph.py and review_standards.py. It imports
and calls the existing modules without modifying them. Selected folders are passed
by reassigning their DOWNLOADS_DIR and EXTRACTED_DIR globals at runtime, using
the same configuration variables as the command-line interfaces.

Extraction results are saved to the project's extracted/ folder.

Resources used when building the app:
- The LangGraph human-in-the-loop UI, replacing input() with on-screen options
  to approve, amend, or request more information, was adapted from:
  https://dev.to/sreeni5018/beyond-input-building-production-ready-human-in-the-loop-ai-with-langgraph-2en9
- The interrupt pattern follows the LangChain documentation:
  https://docs.langchain.com/oss/python/langchain/human-in-the-loop
- The LangGraph debug stream for updates:
  https://sj-langchain.readthedocs.io/en/latest/callbacks/langchain.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html
- The Latvia color theme and Open Sans font are configured in
  .streamlit/config.toml.

Run from the project root:
    pip install streamlit
    ollama serve                       
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st

import extract_graph as pipeline
from extract_graph import DOWNLOADS_DIR, EXTRACTION_MODEL, build_graph, initial_state

try:
    import review_standards as reviewer
    from review_standards import (REVIEW_PROMPT, format_evidence, last_tool_call,
                                  load_extraction, build_reviewer, REVIEW_MODEL,
                                  EXTRACTED_DIR, RUNS_PATH, OLLAMA_NUM_CTX)
    from langchain_ollama import ChatOllama
    from langgraph.types import Command
    REVIEW_OK = True
except Exception:                               # review deps missing: offer extraction only
    REVIEW_OK = False
    REVIEW_MODEL, EXTRACTED_DIR = "mistral-small", Path("extracted")


st.set_page_config(page_title="Latvian procurement extraction", layout="wide")

PREFERRED_MODEL = "gemma4:e4b"          # the extraction choice; preselected if installed
EXTRACT = "Extract procurements"
REVIEW = "Review extracted standards"

# Anchor the default folders to the project directory, so they resolve the same no
# matter where streamlit is launched from (not the current working directory, which
# on a case-insensitive filesystem can drift to the OS Downloads folder).
PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_HOME = PROJECT_DIR / "downloads"
EXTRACTED_HOME = PROJECT_DIR / "extracted"
TABLES_HOME = PROJECT_DIR / "tables"
OCR_HOME = PROJECT_DIR / "ocr"


@st.cache_resource
def get_graph():
    """Compile the extraction graph once and reuse it across reruns."""
    return build_graph()


@st.cache_data(show_spinner=False)
def graph_png():
    """Render the graph with the same draw_mermaid_png call the CLI uses."""
    try:
        return get_graph().get_graph().draw_mermaid_png()
    except Exception:
        return None


def available_models():
    """Return installed Ollama model tags, or [] if Ollama can't be reached."""
    try:
        import ollama
        resp = ollama.list()
        raw = getattr(resp, "models", None) or (resp.get("models", []) if isinstance(resp, dict) else [])
        names = [getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None) for m in raw]
        return sorted(n for n in names if n)
    except Exception:
        return []


def record_from_state(eis_id, state):
    """Assemble the record the pipeline saves (matching extract_graph.process_one)."""
    return {
        "eis_id": eis_id, "source_files": state.get("source_files", []),
        "candidates": state.get("candidates", []), "attempts": state.get("attempts", 0),
        "model_seconds": round(state.get("cpv_seconds", 0.0) + state.get("criteria_seconds", 0.0), 1),
        "extracted": state.get("final"), "critique": state.get("critique"),
        "evaluation_criteria": state.get("criteria"), "criteria_check": state.get("criteria_check"),
        "standards": state.get("standards"),
    }


def save_extraction(eis_id, record, state):
    """Write the same three files the CLI writes: the record, its tables, and its OCR."""
    for home in (EXTRACTED_HOME, TABLES_HOME, OCR_HOME):
        home.mkdir(parents=True, exist_ok=True)
    (EXTRACTED_HOME / f"{eis_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (TABLES_HOME / f"{eis_id}.json").write_text(
        json.dumps(state.get("table_records", []), indent=2, ensure_ascii=False), encoding="utf-8")
    (OCR_HOME / f"{eis_id}.json").write_text(
        json.dumps(state.get("ocr_records", []), indent=2, ensure_ascii=False), encoding="utf-8")
    return EXTRACTED_HOME / f"{eis_id}.json"


def iter_standards(standards):
    """Yield (name, flag) per standard, best-effort across output shapes."""
    if not standards:
        return
    items = standards if isinstance(standards, list) else None
    if items is None and isinstance(standards, dict):
        for key in ("standards", "findings", "schemes", "items", "results"):
            if isinstance(standards.get(key), list):
                items = standards[key]
                break
        if items is None:
            for name, val in standards.items():
                if not str(name).startswith("_"):
                    yield str(name), _std_flag(val)
            return
    for it in items or []:
        if isinstance(it, dict):
            yield str(it.get("name") or it.get("code") or it.get("scheme") or it), _std_flag(it)
        else:
            yield str(it), ""


def _std_flag(val):
    """A short flag for a standard: green and/or phase, if present."""
    if not isinstance(val, dict):
        return ""
    bits = (["green"] if val.get("green") is True else []) + ([str(val["phase"])] if val.get("phase") else [])
    return ", ".join(bits)


def unified_rows(record):
    """Flatten CPV, criteria, and standards into a numbered Field/Value table."""
    extracted = record.get("extracted") or {}
    rows = [{"Field": "Main CPV", "Value": extracted.get("main_cpv") or ""},
            {"Field": "Additional CPV", "Value": ", ".join(extracted.get("additional_cpv") or [])}]
    for i, c in enumerate((record.get("evaluation_criteria") or {}).get("criteria") or [], start=1):
        weight = c.get("weight")
        rows.append({"Field": f"Criterion {i}", "Value": c.get("name") or ""})
        rows.append({"Field": f"Weight {i}", "Value": "" if weight is None else f"{weight:g}"})
    for i, (name, flag) in enumerate(iter_standards(record.get("standards")), start=1):
        rows.append({"Field": f"Standard {i}", "Value": name + (f" ({flag})" if flag else "")})
    return rows


def run_stream(graph, eis_id, log):
    """Run one procurement, drawing every step into a visible placeholder as it happens.

    stream_mode=["updates", "debug"] gives reliable state (updates) plus a task event
    as each node starts and a task_result as it finishes (debug). The growing log is
    drawn into a plain placeholder, not a collapsible container, so the whole trace
    stays on screen while the run is active.
    """
    state, times, started, lines = {}, {}, {}, []
    progress = st.empty()

    def render(note=""):
        progress.markdown("**Progress**\n\n" + ("  \n".join(lines) or "starting") + (f"  \n{note}" if note else ""))

    render()
    run_start = time.perf_counter()
    with redirect_stdout(log):
        for mode, data in graph.stream(initial_state(eis_id), stream_mode=["updates", "debug"]):
            if mode == "updates":
                for _node, writes in (data or {}).items():
                    state.update(writes or {})
                continue
            node = (data.get("payload") or {}).get("name")
            if not node:
                continue
            if data.get("type") == "task":
                started[node] = time.perf_counter()
                lines.append(f"running: {node}")
            elif data.get("type") == "task_result":
                dt = time.perf_counter() - started.get(node, time.perf_counter())
                runs, total = times.get(node, (0, 0.0))
                times[node] = (runs + 1, total + dt)
                lines.append(f"completed: {node} ({dt:.1f}s)")
            else:
                continue
            render()
    wall = time.perf_counter() - run_start
    render(f"done in {wall:.1f}s")
    return state, times, wall


def batch_summary_row(record):
    """One aggregate row per procurement for the batch table."""
    extracted = record.get("extracted") or {}
    ec = record.get("evaluation_criteria") or {}
    return {"id": record["eis_id"], "main_cpv": extracted.get("main_cpv") or "",
            "additional": len(extracted.get("additional_cpv") or []),
            "criteria": len(ec.get("criteria") or []), "reconciled": ec.get("weights_reconcile"),
            "standards": len(list(iter_standards(record.get("standards")))),
            "model_s": record.get("model_seconds"), "error": record.get("error", "")}


# REVIEW (human in the loop)
#
# Reuses review_standards.py's agent, prompt, and interrupt/resume machinery. Only its
# input() turn is replaced by the on-screen decision below, following the UI approach in
# Ramadurai (2025) and LangChain's human-in-the-loop docs (links in the module docstring).

def decision_label(review):
    """The decision word for a verdict: Required, Not required, No decision, or empty."""
    if not review:
        return ""
    applies = review.get("applies")
    if applies is None:
        return "No decision"                        # model called no tool (e.g. a reasoning model)
    return "Required" if applies else "Not required"


def human_note(review):
    """What the person did during review: approved, amended, or sent more information."""
    if not review:
        return ""
    parts = []
    for decision in review.get("human") or []:
        kind = decision.get("type")
        if kind == "approve":
            parts.append("approved")
        elif kind == "edit":
            applies = decision.get("edited_action", {}).get("args", {}).get("applies")
            parts.append(f"amended to {'required' if applies else 'not required'}")
        elif kind == "reject":
            parts.append(f"sent info: {decision.get('message', '')}")
    return "; ".join(parts)


def status_rows(findings, current):
    """Live rows for every candidate: its status now, and the verdict once decided."""
    rows = []
    for i, finding in enumerate(findings):
        review = finding.get("review")
        status = "done" if review else ("reviewing" if i == current else "pending")
        rows.append({"Standard": finding["name"], "Status": status,
                     "Decision": decision_label(review),
                     "Reason": review.get("reason", "") if review else "",
                     "Reviewed": human_note(review)})
    return rows


def final_rows(findings):
    """Standard, category, decision, reason, and what the person did for each candidate."""
    return [{"Standard": f["name"], "Category": f["category"],
             "Decision": decision_label(f.get("review")),
             "Reason": f["review"].get("reason", "") if f.get("review") else "",
             "Reviewed": human_note(f.get("review"))}
            for f in findings]


def review_step():
    """Advance the review by one agent call: start a candidate, or resume after a decision."""
    ss = st.session_state
    finding = ss.rv_findings[ss.rv_index]
    config = {"configurable": {"thread_id": f"standard-{ss.rv_index}"}}
    started = time.perf_counter()
    if not ss.rv_in_finding:
        prompt = REVIEW_PROMPT.format(name=finding["name"], category=finding["category"],
                                      phases=", ".join(finding["phases"]), count=finding["count"],
                                      evidence=format_evidence(finding))
        result = ss.rv_agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                                    config=config, version="v2")
        ss.rv_in_finding, ss.rv_seconds, ss.rv_pauses, ss.rv_proposed, ss.rv_human = True, 0.0, 0, None, []
    else:
        ss.rv_human.append(ss.rv_decision)
        result = ss.rv_agent.invoke(Command(resume={"decisions": [ss.rv_decision]}),
                                    config=config, version="v2")
        ss.rv_decision = None
    ss.rv_seconds += time.perf_counter() - started

    if result.interrupts:
        request = result.interrupts[0].value["action_requests"][0]
        if ss.rv_proposed is None:
            ss.rv_proposed = request["args"].get("applies")
        ss.rv_pauses += 1
        ss.rv_request = request
        ss.rv_awaiting = True
        return

    verdict = last_tool_call(result) or {"applies": None,
                                         "reason": "No decision: the model returned no tool call."}
    for decision in reversed(ss.rv_human):          # a human amend is the final word
        if decision.get("type") == "edit":
            edited = decision["edited_action"]["args"]
            verdict["applies"] = edited.get("applies")
            verdict["reason"] = edited.get("reason") or verdict.get("reason")
            break
    verdict["seconds"] = round(ss.rv_seconds, 1)
    verdict["pauses"] = ss.rv_pauses
    verdict["proposed"] = verdict.get("applies") if ss.rv_proposed is None else ss.rv_proposed
    verdict["overridden"] = verdict["proposed"] != verdict.get("applies")
    verdict["human"] = ss.rv_human
    finding["review"] = verdict
    ss.rv_index, ss.rv_in_finding, ss.rv_awaiting = ss.rv_index + 1, False, False


def render_decision(finding, request):
    """Show one paused candidate and collect the person's decision (approve, amend, or send more information)."""
    args = request.get("args", {})
    applies = args.get("applies")
    st.warning(f"Review needed: {finding['name']} ({finding['category']})")
    st.write(f"Appears in {', '.join(finding['phases'])}, mentioned {finding['count']} times.")
    for snippet in finding["evidence"]:
        st.caption(f"({snippet['phase']}) {snippet['text']}")
    st.markdown(f"**Model's verdict:** this **{'IS' if applies else 'is NOT'}** a required standard. "
                f"{args.get('reason', '')}")
    st.caption("Approve keeps the model's verdict. Amend replaces it with your own decision and "
               "reason, final. Send more information hands your note to the model to reconsider.")
    choice = st.radio("Your decision",
                      ["Approve", "Amend with your own notes", "Send the model more information"],
                      key="rv_choice")
    new_applies, note, info = applies, "", ""
    if choice == "Amend with your own notes":
        new_applies = st.radio("It is", ["a required standard", "not a required standard"],
                               index=0 if applies else 1, key="rv_new") == "a required standard"
        note = st.text_input("Your reason", key="rv_note")
    if choice == "Send the model more information":
        info = st.text_input("What should the model consider?", key="rv_info")
    if st.button("Submit decision", type="primary"):
        if choice == "Approve":
            decision = {"type": "approve"}
        elif choice == "Amend with your own notes":
            decision = {"type": "edit",
                        "edited_action": {"name": "request_review",
                                          "args": {**args, "applies": new_applies,
                                                   "reason": note or args.get("reason")}}}
        else:
            decision = {"type": "reject", "message": info or "reconsider this candidate"}
        st.session_state.rv_decision = decision
        st.session_state.rv_awaiting = False
        st.rerun()


def save_review(path, record, findings, model_tag):
    """Write the verdicts back into the extracted file and append the run log."""
    reviewed = [f for f in findings if f.get("review")]
    kept = [f for f in reviewed if f["review"].get("applies")]
    seconds = sum(f["review"]["seconds"] for f in reviewed)
    summary = {"model": model_tag, "candidates": len(findings), "applies": len(kept),
               "pauses": sum(f["review"]["pauses"] for f in reviewed),
               "overridden": sum(1 for f in reviewed if f["review"]["overridden"]),
               "seconds": round(seconds, 1),
               "seconds_each": round(seconds / len(findings), 1) if findings else 0.0}
    record["standards_review"] = summary
    record["standards_reviewed"] = True
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(RUNS_PATH, "a", encoding="utf-8") as runs:
        runs.write(json.dumps({**summary, "eis_id": record.get("eis_id"),
                               "verdicts": {f["name"]: f["review"] for f in reviewed}},
                              ensure_ascii=False) + "\n")


# SIDEBAR

st.sidebar.header("Settings")
settings = st.sidebar.container()          # filled once the action is known
st.sidebar.divider()
options = [EXTRACT] + ([REVIEW] if REVIEW_OK else [])
action = st.sidebar.selectbox("Select action", options)
if not REVIEW_OK:
    st.sidebar.caption("Review is unavailable (its dependencies did not import).")

group = "review" if action == REVIEW else "extract"
default_folder = EXTRACTED_HOME if group == "review" else DOWNLOADS_HOME
with settings:
    default_model = REVIEW_MODEL if group == "review" else PREFERRED_MODEL
    models = available_models()
    if models:
        model = st.selectbox("Ollama model", models, key=f"model_{group}",
                             index=models.index(default_model) if default_model in models else 0)
    else:
        model = st.text_input("Ollama model", value=default_model, key=f"model_{group}")
    # Reset the folder to the action's default (downloads/ or extracted/) whenever the
    # action changes; within an action a typed path is kept.
    if st.session_state.get("last_action") != action:
        st.session_state["folder_box"] = str(default_folder)
        st.session_state["last_action"] = action
    folder = Path(st.text_input("Folder", key="folder_box"))
    if not folder.is_dir():                          # invalid path: fall back to the default
        st.caption(f"{folder} is not a folder; using {default_folder}")
        folder = default_folder
    if group == "extract":
        pipeline.DOWNLOADS_DIR = folder            # the graph reads its documents from here

if action == REVIEW:
    ids = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
else:
    ids = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
if not ids:
    st.sidebar.warning(f"Nothing to work on under {folder}/")

# One multiselect for both actions: leave it empty to mean every id in the folder.
chosen = st.sidebar.multiselect("Procurement ids (leave empty for all)", ids, key=f"sel_{group}")
selected = chosen or ids

run = start_review = None
if action == EXTRACT:
    run = st.sidebar.button(f"Run extraction ({len(selected)})", type="primary", disabled=not ids)
else:
    start_review = st.sidebar.button(f"Start review ({len(selected)})", type="primary", disabled=not ids)


def ensure_model():
    """Rebind the extraction models if the chosen tag changed since the last run."""
    if model != st.session_state.get("model"):
        with st.spinner(f"Loading model {model}..."):
            pipeline.build_models(model)
        st.session_state["model"] = model


def load_review_id(eis_id):
    """Load one procurement's candidates for review, discarding any prior verdicts."""
    path, record, findings = load_extraction(eis_id)
    for finding in findings:
        finding.pop("review", None)
    st.session_state.update(rv_path=str(path), rv_record=record, rv_findings=findings,
                            rv_index=0, rv_in_finding=False, rv_awaiting=False,
                            rv_decision=None, rv_id=eis_id, rv_human=[])


# MAIN

st.title("Local AI to extract Latvian public procurement data")
st.caption("Pick a folder, a model, and an action on the left. Leave the id list empty to work on the "
           "whole folder. Extractions are saved to extracted/.")

if run and selected:
    ensure_model()
    for key in ("record", "batch", "log", "timings", "wall", "cpv_s", "criteria_s",
                "reading", "tables", "ocr", "saved"):
        st.session_state.pop(key, None)
    if len(selected) == 1:
        eis_id = selected[0]
        log = io.StringIO()
        try:
            state, times, wall = run_stream(get_graph(), eis_id, log)
        except Exception as error:
            st.error(f"Pipeline failed: {error}")
            st.stop()
        record = record_from_state(eis_id, state)
        saved_path = save_extraction(eis_id, record, state)
        st.session_state.update(record=record, saved=str(saved_path), log=log.getvalue(),
                                timings=times, wall=wall, cpv_s=state.get("cpv_seconds", 0.0),
                                criteria_s=state.get("criteria_seconds", 0.0),
                                tables=state.get("table_records", []), ocr=state.get("ocr_records", []),
                                reading={"chars": len(state.get("documents_text", "")),
                                         "files": len(state.get("source_files", [])),
                                         "tables": len(state.get("tables", []))})
    else:
        graph = get_graph()
        records = []
        progress = st.progress(0.0, text="Starting")
        with redirect_stdout(io.StringIO()):
            for i, eid in enumerate(selected):
                progress.progress(i / len(selected), text=f"Running {eid} ({i + 1}/{len(selected)})")
                try:
                    eid_state = graph.invoke(initial_state(eid))
                    eid_record = record_from_state(eid, eid_state)
                    save_extraction(eid, eid_record, eid_state)
                    records.append(eid_record)
                except Exception as error:
                    records.append({"eis_id": eid, "error": str(error), "extracted": None,
                                    "evaluation_criteria": None, "standards": None})
        progress.progress(1.0, text=f"Done: {len(selected)} procurements")
        st.session_state["batch"] = records

if start_review and selected:
    reviewer.EXTRACTED_DIR = folder
    if model != st.session_state.get("review_model"):
        with st.spinner(f"Loading model {model}..."):
            st.session_state.review_agent = build_reviewer(
                ChatOllama(model=model, temperature=0, num_ctx=OLLAMA_NUM_CTX))
        st.session_state.review_model = model
    for k in [k for k in st.session_state if k.startswith("rv_")]:
        st.session_state.pop(k, None)
    st.session_state.update(rv_active=True, rv_queue=list(selected), rv_qpos=0,
                            rv_model=model, rv_agent=st.session_state.review_agent, rv_done=[])
    load_review_id(selected[0])
    st.rerun()


# DISPLAY

if action == REVIEW:
    reviewer.EXTRACTED_DIR = folder
    if not st.session_state.get("rv_active"):
        st.info("Pick ids to review (leave empty for all) and click Start review. This re-reviews the "
                "selected procurements and overwrites their verdicts.")
        st.stop()

    queue = st.session_state.rv_queue
    if st.session_state.rv_done:
        st.caption(f"Reviewed {len(st.session_state.rv_done)} of {len(queue)} so far.")
        st.dataframe(st.session_state.rv_done, use_container_width=True, hide_index=True)

    if st.session_state.rv_qpos >= len(queue):
        st.success(f"Review queue complete: {len(queue)} procurement(s), verdicts saved to their files.")
        if st.button("Review another set"):
            for k in [k for k in st.session_state if k.startswith("rv_")]:
                st.session_state.pop(k, None)
            st.rerun()
        st.stop()

    findings = st.session_state.rv_findings
    current_id = st.session_state.rv_id
    st.subheader(f"Reviewing {current_id} ({st.session_state.rv_qpos + 1} of {len(queue)})")

    # This id's candidates are all decided: show them, then save and step the queue on.
    if st.session_state.rv_index >= len(findings):
        kept = sum(1 for f in findings if f.get("review") and f["review"].get("applies"))
        st.caption(f"Kept {kept} of {len(findings)} as required standards.")
        st.dataframe(final_rows(findings), use_container_width=True, hide_index=True)
        last = st.session_state.rv_qpos + 1 >= len(queue)
        if st.button("Save and finish" if last else "Save and review the next procurement",
                     type="primary"):
            if findings:
                save_review(Path(st.session_state.rv_path), st.session_state.rv_record,
                            findings, st.session_state.rv_model)
            st.session_state.rv_done.append({"id": current_id, "candidates": len(findings), "kept": kept})
            st.session_state.rv_qpos += 1
            if not last:
                load_review_id(queue[st.session_state.rv_qpos])
            st.rerun()
        st.stop()

    st.dataframe(status_rows(findings, st.session_state.rv_index),
                 use_container_width=True, hide_index=True)
    # One placeholder holds either the decision screen or the next-candidate line, so
    # the decision widgets are replaced cleanly instead of lingering greyed out.
    area = st.empty()
    if st.session_state.rv_awaiting:
        with area.container():
            render_decision(findings[st.session_state.rv_index], st.session_state.rv_request)
        st.stop()
    with area.container():
        st.info(f"Candidate {st.session_state.rv_index + 1} of {len(findings)}: "
                f"{findings[st.session_state.rv_index]['name']}")
    try:
        review_step()
    except Exception as error:
        st.error(f"Review failed: {error}")
        st.stop()
    st.rerun()

# EXTRACT results: an aggregate table for several ids, the detailed view for one.
batch = st.session_state.get("batch")
if batch:
    st.subheader(f"Batch results ({len(batch)} procurements)")
    st.caption(f"Saved to {EXTRACTED_HOME}/ (and tables/, ocr/).")
    st.dataframe([batch_summary_row(r) for r in batch], use_container_width=True, hide_index=True)
    failed = [r for r in batch if r.get("error")]
    if failed:
        st.warning(f"{len(failed)} procurement(s) failed; see the error column.")
    st.stop()

record = st.session_state.get("record")
if not record:
    st.info("Pick ids (leave empty for all) and click Run extraction.")
    st.stop()

st.subheader("Extracted data")
if st.session_state.get("saved"):
    st.caption(f"Saved to {st.session_state['saved']} (and tables/, ocr/).")
st.dataframe(unified_rows(record), use_container_width=True, hide_index=True)

st.subheader("Pipeline")
png = graph_png()
if png:
    st.image(png)
else:
    st.caption("Could not render the graph. draw_mermaid_png needs internet (mermaid.ink).")

st.markdown("**Statistics on processed documents**")
reading = st.session_state.get("reading") or {}
st.caption(f"{reading.get('chars', 0):,} characters from {reading.get('files', 0)} files, "
           f"{reading.get('tables', 0)} tables")

st.subheader("Timing")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Wall clock", f"{st.session_state.get('wall', 0):.1f}s")
t2.metric("CPV model time", f"{st.session_state.get('cpv_s', 0):.1f}s")
t3.metric("Criteria model time", f"{st.session_state.get('criteria_s', 0):.1f}s")
t4.metric("Model total runtime", f"{record.get('model_seconds', 0)}s")
times = st.session_state.get("timings") or {}
if times:
    st.dataframe([{"Node": n, "Times run": runs, "Seconds": round(secs, 1)}
                  for n, (runs, secs) in sorted(times.items(), key=lambda kv: kv[1][1], reverse=True)],
                 use_container_width=True, hide_index=True)

with st.expander("Full JSON record"):
    st.json(record)
if st.session_state.get("log"):
    with st.expander("Run log"):
        st.code(st.session_state["log"])

st.subheader("Inspect extracted tables and OCR outputs")
with st.expander("Extracted tables (per file)"):
    st.json(st.session_state.get("tables", []))
with st.expander("OCR outputs (per scanned file)"):
    st.json(st.session_state.get("ocr", []))