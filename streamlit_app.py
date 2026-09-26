"""Streamlit interface for procurement extraction and standards review.

Run: streamlit run streamlit_app.py
Extraction uses extract_graph.py; review decisions use review_standards.py.
Results are saved to extracted/, with supporting tables/ and ocr/ files.

Human-review UI reference:
https://dev.to/sreeni5018/beyond-input-building-production-ready-human-in-the-loop-ai-with-langgraph-2en9
Interrupts: https://docs.langchain.com/oss/python/langchain/human-in-the-loop
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st

import extract_graph as pipeline
from extract_graph import build_graph, initial_state

try:
    from review_standards import (ReviewSession, make_decision, save_review, review_summary,
                                  load_extraction, build_reviewer, REVIEW_MODEL, OLLAMA_NUM_CTX)
    from review_context import load_support, evidence_rows, print_context
    from langchain_ollama import ChatOllama
    REVIEW_OK = True
except ImportError as error:
    REVIEW_OK = False
    REVIEW_ERROR = str(error)
    REVIEW_MODEL = "gemma4:e4b"


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


def choose_folder(default_folder, group):
    """Browse local folders and remember a separate location for each action."""
    key = f"browse_{group}"
    if key not in st.session_state:
        st.session_state[key] = str(default_folder if default_folder.is_dir() else PROJECT_DIR)
    folder = Path(st.session_state[key])
    locked = group == "review" and st.session_state.get("rv_active", False)
    st.write("Selected folder")
    st.code(str(folder), language=None)
    if locked:
        st.caption("Finish the current review session before changing its folder.")
        return folder
    up, home = st.columns(2)
    if up.button("Up one level", key=f"{key}_up", disabled=folder == folder.parent):
        st.session_state[key] = str(folder.parent)
        st.rerun()
    if home.button("Project folder", key=f"{key}_home"):
        st.session_state[key] = str(PROJECT_DIR)
        st.rerun()
    try:
        children = sorted(p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.'))
    except OSError as error:
        st.warning(f"Cannot browse this folder: {error}")
        children = []
    if children:
        child = st.selectbox("Subfolders", children, key=f"{key}_child_{folder}")
        if st.button("Open selected subfolder", key=f"{key}_open"):
            st.session_state[key] = str(folder / child)
            st.rerun()
    else:
        st.caption("No visible subfolders. Use this folder or go up one level.")
    with st.expander("Enter a path manually"):
        typed = st.text_input("Folder path", value=str(folder), key=f"{key}_path_{folder}")
        if st.button("Use this path", key=f"{key}_use"):
            candidate = Path(typed).expanduser()
            if not candidate.is_absolute():
                candidate = PROJECT_DIR / candidate
            if candidate.is_dir():
                st.session_state[key] = str(candidate.resolve())
                st.rerun()
            else:
                st.error("Folder not found. The selected folder has not changed.")
    return folder

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


def iter_standards(standards):
    """Read the candidate list saved by the extraction pipeline."""
    for finding in standards or []:
        yield finding["name"], ", ".join(finding.get("phases") or [])


def unified_rows(record):
    """Flatten CPV, criteria, and standards into a numbered Field/Value table."""
    extracted = record.get("extracted") or {}
    rows = [{"Field": "Main CPV", "Value": extracted.get("main_cpv") or ""},
            {"Field": "Additional CPV", "Value": ", ".join(extracted.get("additional_cpv") or [])}]
    ec = record.get("evaluation_criteria") or {}
    for i, c in enumerate(ec.get("criteria") or [], start=1):
        scope = " (all lots)" if ec.get("same_for_all_lots") else (f" (lot {c['lot']})" if c.get("lot") else "")
        weight = c.get("weight")
        rows.append({"Field": f"Criterion {i}{scope}", "Value": c.get("name") or ""})
        rows.append({"Field": f"Weight {i}{scope}", "Value": "" if weight is None else f"{weight:g}"})
    for i, (name, flag) in enumerate(iter_standards(record.get("standards")), start=1):
        rows.append({"Field": f"Standard {i}", "Value": name + (f" ({flag})" if flag else "")})
    return rows


def run_stream(graph, eis_id, log, downloads_dir):
    """Run one procurement, drawing every step into a visible placeholder as it happens."""
    state, times, started, lines = {}, {}, {}, []
    progress = st.empty()

    def render(note=""):
        progress.markdown("**Progress**\n\n" + ("  \n".join(lines) or "starting") + (f"  \n{note}" if note else ""))

    render()
    run_start = time.perf_counter()
    with redirect_stdout(log):
        for mode, data in graph.stream(initial_state(eis_id, downloads_dir), stream_mode=["updates", "debug"]):
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
    """Display a completed decision or a pending review."""
    if not review:
        return "Pending"
    if review.get("status") == "unresolved" or not isinstance(review.get("applies"), bool):
        return "Needs human review"
    return "Required" if review["applies"] else "Not required"


def status_rows(findings, current=None):
    """Show decisions, their source and the person's actions."""
    rows = []
    for i, finding in enumerate(findings):
        review = finding.get("review") or {}
        human = review.get("human") or []
        source = review.get("decision_source")
        if not source and review:
            source = "human" if human and human[-1].get("type") in {"approve", "edit"} else "model"
        actions = {"approve": "Approved", "edit": "Supplied decision", "reject": "Sent information"}
        rows.append({"Standard": finding["name"], "Category": finding["category"],
                     "Decision": decision_label(review), "Decision by": source or "",
                     "Human actions": "; ".join(actions.get(d.get("type"), "") for d in human),
                     "Reason": review.get("reason", ""),
                     "Current candidate": i == current})
    return rows


def review_step():
    """Advance one candidate and wait whenever human input is needed."""
    ss = st.session_state
    finding = ss.rv_findings[ss.rv_index]
    if ss.rv_session is None:
        ss.rv_session = ReviewSession(ss.rv_agent, finding)
        ss.rv_session.start()
    elif ss.rv_decision is not None:
        decision, ss.rv_decision = ss.rv_decision, None
        ss.rv_session.answer(decision)
    if ss.rv_session.request:
        ss.rv_request = ss.rv_session.request
        ss.rv_awaiting = True
        return
    if ss.rv_session.verdict is not None:
        finding["review"] = ss.rv_session.verdict
        ss.rv_index += 1
        ss.rv_session = None
        ss.rv_awaiting = False
        for key in ("rv_choice", "rv_new", "rv_note", "rv_info"):
            ss.pop(key, None)


def render_decision(finding, request):
    """Present the evidence and collect a human decision or more information."""
    ss = st.session_state
    args = request["args"]
    applies = args.get("applies")
    st.warning(f"Review needed: {finding['name']}")
    for snippet in evidence_rows(finding):
        st.write(f"{snippet['phase']}: {snippet['text']}")
    with st.expander("More evidence and procurement context"):
        context = io.StringIO()
        with redirect_stdout(context):
            matches = print_context(ss.rv_record, finding, ss.rv_support, ss.rv_session.human)
        st.text(context.getvalue())
        for block in matches:
            st.caption(f"{block['source']} | {block['location']}")
            st.text(block["text"])
        st.write("Procurement source files", ss.rv_record.get("source_files") or [])
    st.write(args.get("reason", ""))
    options = ["Decide", "Send more information"]
    if isinstance(applies, bool):
        st.write("Model decision:", "Required" if applies else "Not required")
        options.insert(0, "Approve")
    else:
        st.info("No model decision is available. Make a decision or supply evidence for another attempt.")
    choice = st.radio("Your action", options, key="rv_choice")
    if choice == "Decide":
        value = st.radio("Is this required?", ["Required", "Not required"], index=None, key="rv_new")
        reason = st.text_area("Reason", key="rv_note")
    elif choice == "Send more information":
        note = st.text_area("Evidence or clarification for the model", key="rv_info")
    if st.button("Submit decision", type="primary"):
        if choice == "Approve":
            decision = {"type": "approve"}
        elif choice == "Decide":
            if value is None or not reason.strip():
                st.warning("Select a decision and enter a reason.")
                return
            decision = make_decision(request, value == "Required", reason.strip())
        else:
            if not note.strip():
                st.warning("Enter evidence or clarification.")
                return
            decision = {"type": "reject", "message": note.strip()}
        ss.rv_decision, ss.rv_awaiting = decision, False
        for key in ("rv_choice", "rv_new", "rv_note", "rv_info"):
            ss.pop(key, None)
        st.rerun()


# SIDEBAR

st.sidebar.header("Settings")
settings = st.sidebar.container()          # filled once the action is known
st.sidebar.divider()
options = [EXTRACT] + ([REVIEW] if REVIEW_OK else [])
action = st.sidebar.selectbox("Select action", options)
if not REVIEW_OK:
    st.sidebar.error(f"Review is unavailable: {REVIEW_ERROR}")

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
    folder = choose_folder(default_folder, group)

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


def load_review_id(eis_id, extracted_dir):
    """Load one procurement's candidates for review, discarding any prior verdicts."""
    path, record, findings = load_extraction(eis_id, extracted_dir)
    for finding in findings:
        finding.pop("review", None)
    st.session_state.update(rv_path=str(path), rv_record=record, rv_findings=findings,
                            rv_index=0, rv_session=None, rv_awaiting=False,
                            rv_decision=None, rv_id=eis_id, rv_support=load_support(path))


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
            state, times, wall = run_stream(get_graph(), eis_id, log, folder)
        except Exception as error:
            st.error(f"Pipeline failed: {error}")
            st.stop()
        record = pipeline.build_record(eis_id, state)
        saved_path, _, _ = pipeline.save_result(eis_id, state, EXTRACTED_HOME, TABLES_HOME, OCR_HOME)
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
                    eid_state = graph.invoke(initial_state(eid, folder))
                    # eid_record = record_from_state(eid, eid_state)
                    # save_extraction(eid, eid_record, eid_state)
                    eid_record = pipeline.build_record(eid, eid_state)
                    pipeline.save_result(eid, eid_state, EXTRACTED_HOME, TABLES_HOME, OCR_HOME)
                    records.append(eid_record)
                except Exception as error:
                    records.append({"eis_id": eid, "error": str(error), "extracted": None,
                                    "evaluation_criteria": None, "standards": None})
        progress.progress(1.0, text=f"Done: {len(selected)} procurements")
        st.session_state["batch"] = records

if start_review and selected:
    if model != st.session_state.get("review_model"):
        with st.spinner(f"Loading model {model}..."):
            st.session_state.review_agent = build_reviewer(
                ChatOllama(model=model, temperature=0, num_ctx=OLLAMA_NUM_CTX))
        st.session_state.review_model = model
    for k in [k for k in st.session_state if k.startswith("rv_")]:
        st.session_state.pop(k, None)
    st.session_state.update(rv_active=True, rv_queue=list(selected), rv_qpos=0,
                            rv_model=model, rv_agent=st.session_state.review_agent, rv_done=[], rv_folder=str(folder))
    load_review_id(selected[0], folder)
    st.rerun()


# DISPLAY

if action == REVIEW:
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

    # Save the procurement after every candidate has a decision.
    if st.session_state.rv_index >= len(findings):
        summary = review_summary(findings, st.session_state.rv_model)
        st.caption(f"Decided {summary['decided']}; unresolved {summary['unresolved']}; "
                   f"human reviewed {summary['human_reviewed']}; "
                   f"kept {summary['applies']} of {summary['candidates']} as required standards.")
        st.dataframe(status_rows(findings), use_container_width=True, hide_index=True)
        last = st.session_state.rv_qpos + 1 >= len(queue)
        if st.button("Save and finish" if last else "Save and review the next procurement",
                     type="primary"):
            save_review(Path(st.session_state.rv_path), st.session_state.rv_record,
                            findings, st.session_state.rv_model)
            st.session_state.rv_done.append({
                "Procurement ID": current_id,
                "Candidates": len(findings),
                "Required": summary["applies"],
                "Human-reviewed candidates": summary["human_reviewed"],
                "Human responses": sum(len((f.get("review") or {}).get("human") or [])
                                       for f in findings),
            })
            st.session_state.rv_qpos += 1
            if not last:
                load_review_id(queue[st.session_state.rv_qpos], Path(st.session_state.rv_folder))
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
