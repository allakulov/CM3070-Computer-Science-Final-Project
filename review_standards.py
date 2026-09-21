"""Review the standards found in a procurement, with a human in the loop.

This is step two, run after extract_graph.py has finished. Extraction is slow, so
it runs unattended; reviewing needs a person at the keyboard, and separating the
two means nobody has to sit waiting for documents to be read. The candidates are
read back from extracted/{eis_id}.json rather than found again, so the documents
are never re-opened.

standards.py finds candidates by pattern alone, so it cannot tell whether the
surrounding prose actually imposes a standard. A notice may name a rule only to
say it does not apply ("neattiecas"), or in boilerplate that has nothing to do
with what is being bought. This module reads the evidence for each candidate and
decides whether it really is a required or referenced standard.

The agent has two tools and chooses between them. A clear case goes to
record_decision and is applied without stopping. An ambiguous case goes to
request_review, which the human-in-the-loop middleware interrupts on, so a person
sees the evidence and approves the model's verdict, amends it with their own
decision and reason, or sends the model more information to reconsider. An amend is
the reviewer's final word; sending information lets the model decide again.

The outcome is written back into the same extracted file, added to each candidate
rather than replacing it, so the pattern pass stays visible next to the verdict.

Run:
    python review_standards.py --id 123450
"""

import argparse
import json
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# CONFIGURATION

EXTRACTED_DIR = Path("extracted")    # where extract_graph.py saves its results
RUNS_PATH = Path("standards_review_runs.jsonl")   # one line per run, for comparing models
REVIEW_MODEL = "gemma4:e4b"        
OLLAMA_NUM_CTX = 4096                # Ollama's default of 2048 is too small


REVIEW_PROMPT = """You are reading one candidate standard or certificate found in a \
Latvian public procurement notice. It was found by pattern matching, so it may be a \
false match. Along with the candidate, you are provided evidence in the form of \
text surrounding the pattern that was matched. This evidence is the main basis for\
your evaluation, not the name of the candidate standard or certification scheme.

Decide whether this is genuinely a standard, certificate or certification scheme that \
the procurement requires or references for what is being bought.

Set applies to false when:
- the evidence shows the rule does not apply (Latvian "neattiecas");
- it is a general legal obligation governing how the contract is performed rather than \
something a bidder must hold or a product must meet, for example data protection \
(GDPR, regulation 2016/679), employment law, or property and tenancy rules;
- the standard is named only as an example, or in unrelated boilerplate.

Set applies to true only when a bidder, a product or a service must meet or hold it.

Call record_decision when the evidence is clear either way. Call request_review when \
the evidence is ambiguous or too short, meaning it doesn't contain enough relevant information. \
Whichever tool you call, give your best judgement based on the evidence prose.

CANDIDATE: {name}
CATEGORY: {category}
APPEARS IN: {phases} (mentioned {count} times in total)
EVIDENCE, one passage per place it appears:
{evidence}
"""


# TOOLS
#
# The two tools differ only in whether a person sees them: the middleware is told to
# interrupt on request_review and to let record_decision through. Choosing a tool is
# therefore how the model says whether it is confident.

# return_direct stops the agent as soon as the tool runs. Without it the tool result
# goes back to the model, which has nothing left to do and simply calls the same tool
# again, so the same candidate is put to the reviewer over and over.
@tool(return_direct=True, response_format="content_and_artifact")
def record_decision(name: str, applies: bool, reason: str) -> tuple[str, dict]:
    """Record a clear decision about a candidate standard.

    Use only when the evidence is unambiguous.

    Args:
        name: The candidate standard or certificate.
        applies: True if it is genuinely required or referenced here.
        reason: One short sentence of justification.
    """
    receipt = {"decision_recorded": True, "name": name, "applies": applies, "reason": reason}
    return json.dumps(receipt), receipt


@tool(return_direct=True, response_format="content_and_artifact")
def request_review(name: str, applies: bool, reason: str) -> tuple[str, dict]:
    """Ask a person to decide about a candidate standard.

    Use when the evidence is ambiguous or doesn't contain enough information.

    Args:
        name: The candidate standard or certificate.
        applies: Your best guess, for the reviewer to confirm.
        reason: One short sentence on why this is unclear.
    """
    receipt = {"decision_recorded": True, "name": name, "applies": applies, "reason": reason}
    return json.dumps(receipt), receipt


# AGENT

def build_reviewer(model=None):
    """Build the review agent with the human-in-the-loop middleware.

    interrupt_on maps a tool to whether a person must see it: record_decision runs
    straight through, request_review pauses. The reviewer can approve the verdict,
    amend it (edit, authoritative), or send more information (reject) for the model to
    reconsider. The checkpointer saves the run so it can resume after the person answers.
    """
    model = model or ChatOllama(model=REVIEW_MODEL, temperature=0, num_ctx=OLLAMA_NUM_CTX)
    return create_agent(
        model=model,
        tools=[record_decision, request_review],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "record_decision": False,
                    "request_review": {"allowed_decisions": ["approve", "edit", "reject"]},
                },
                description_prefix="A candidate standard needs review",
            ),
        ],
        checkpointer=InMemorySaver(),
    )


# HUMAN TURN

def ask_person(finding, request):
    """Show one paused candidate and return the person's decision.

    The reviewer approves the model's verdict, amends it with their own decision and
    reason, or sends the model more information to reconsider. The thing being judged
    is the verdict, not the candidate, so the prompt shows what the model concluded.

    Args:
        finding (dict): The candidate read from the extracted file.
        request (dict): The action the middleware paused on.

    Returns:
        dict: A decision for Command(resume=...).
    """
    arguments = request.get("args", {})
    applies = arguments.get("applies")
    verdict = "IS a required standard" if applies else "is NOT a required standard"

    print("\n  review needed")
    print(f"    candidate: {finding['name']} ({finding['category']})")
    print(f"    appears in: {', '.join(finding['phases'])}, {finding['count']} times in total")
    print("    evidence:")
    for snippet in finding["evidence"]:
        print(f"      ({snippet['phase']}) {snippet['text']}")
    print(f"    the model's verdict: this {verdict}, because {arguments.get('reason')}")
    print("    decide:  a approve the verdict   m amend it with your own decision and reason   "
          "i send the model more information to reconsider")
    while True:
        answer = input("    [a/m/i]: ").strip().lower()
        if answer in {"a", "m", "i"}:
            break
        print("    Enter a, m or i.")

    if answer == "m":
        while True:
            answer = input("    is it a required standard? [y/n]: ").strip().lower()
            if answer in {"y", "n"}:
                break
            print("    Enter y or n.")
        chosen = answer == "y"
        note = input("    your reason: ").strip()
        return {"type": "edit",
                "edited_action": {"name": "request_review",
                                  "args": {**arguments, "applies": chosen,
                                           "reason": note or arguments.get("reason")}}}
    if answer == "i":
        return {"type": "reject", "message": input("    what should the model consider: ").strip()}
    return {"type": "approve"}


# REVIEW

def format_evidence(finding):
    """Lay out every evidence passage for one candidate, labelled by phase."""
    return "\n".join(f"- ({snippet['phase']}) {snippet['text']}"
                     for snippet in finding["evidence"])


def timed_invoke(agent, message, config):
    """Run the agent once and return its result and how long the model took.

    Timing each call separately is what keeps a person's thinking time out of the
    measurement: the clock only runs while the agent is working.
    """
    started = time.perf_counter()
    result = agent.invoke(message, config=config, version="v2")
    return result, time.perf_counter() - started


def review_finding(agent, finding, index):
    """Run the agent over one candidate, pausing for a person when it asks to.

    Returns:
        dict: The review outcome, or None if the agent called no tool.
    """
    config = {"configurable": {"thread_id": f"standard-{index}"}}
    prompt = REVIEW_PROMPT.format(name=finding["name"], category=finding["category"],
                                  phases=", ".join(finding["phases"]),
                                  count=finding["count"],
                                  evidence=format_evidence(finding))
    result, seconds = timed_invoke(agent, {"messages": [{"role": "user", "content": prompt}]}, config)
    pauses = 0
    proposed = None
    human = []

    # the middleware pauses by raising an interrupt; answer it and resume.
    while result.interrupts:
        pauses += 1
        decisions = []
        for interrupt in result.interrupts:
            for request in interrupt.value["action_requests"]:
                if proposed is None:
                    # what the model itself suggested, before any correction from me
                    proposed = request["args"].get("applies")
                decision = ask_person(finding, request)
                human.append(decision)
                decisions.append(decision)
        result, extra = timed_invoke(agent, Command(resume={"decisions": decisions}), config)
        seconds += extra

    return review_outcome(result, finding["name"], human, seconds, pauses, proposed)


def review_outcome(result, expected_name, human, seconds, pauses, proposed):
    """Shared CLI/app decision policy, including authoritative final amendments.

    Only the last human action can override a completed receipt. If no receipt
    exists, retain the edit for inspection but do not claim successful execution.
    """
    verdict = last_tool_call(result, expected_name=expected_name)
    completed = verdict is not None
    verdict = dict(verdict) if completed else {
        "tool": None, "applies": None, "reason": "No completed decision after review."}
    verdict["status"] = "decided" if completed else "unresolved"
    if human and human[-1].get("type") == "edit":
        edited_action = human[-1].get("edited_action") or {}
        edited = edited_action.get("args") or {}
        if (edited_action.get("name") in {"record_decision", "request_review"}
                and edited.get("name") == expected_name
                and isinstance(edited.get("applies"), bool)
                and isinstance(edited.get("reason"), str)):
            authoritative = {key: edited[key] for key in ("applies", "reason")}
            if completed:
                received = {key: verdict.get(key) for key in ("applies", "reason")}
                if received != authoritative:
                    verdict["amend_discrepancy"] = {"receipt": received, "human_edit": authoritative}
                    verdict.update(authoritative)
                    verdict["decision_source"] = "human_edit_reconciled"
                else:
                    verdict["decision_source"] = "executed_receipt"
            else:
                verdict["pending_human_edit"] = authoritative
    verdict["human"] = human
    verdict["seconds"] = round(seconds, 1)
    verdict["pauses"] = pauses
    verdict["proposed"] = verdict["applies"] if proposed is None else proposed
    verdict["overridden"] = verdict["applies"] is not None and verdict["proposed"] != verdict["applies"]
    return verdict


def last_tool_call(result, expected_name=None):
    """Accept only a recognised decision with matching successful tool output.

    An AI tool proposal is not a completed decision. A later unresolved proposal
    supersedes earlier completed calls. New outputs carry the executed arguments;
    exact legacy acknowledgements remain supported for existing histories.
    """
    messages = result.value["messages"]
    for index in range(len(messages)-1, -1, -1):
        calls = getattr(messages[index], "tool_calls", []) or []
        for call in reversed(calls):
            if call.get("name") not in {"record_decision", "request_review"}:
                continue
            args = call.get("args") or {}
            replies = [m for m in messages[index+1:]
                       if getattr(m, "type", None) == "tool"
                       and getattr(m, "tool_call_id", None) == call.get("id")]
            if not replies or getattr(replies[-1], "status", "success") == "error":
                return None
            content = replies[-1].content
            data = getattr(replies[-1], "artifact", None)
            if data is None:
                try:
                    data = json.loads(content)
                except (TypeError, ValueError):
                    prefix = "recorded" if call["name"] == "record_decision" else "reviewed"
                    expected = f"{prefix} {args.get('name')}: applies={args.get('applies')}"
                    if content != expected:
                        return None
                    data = {**args, "decision_recorded": True}
            if not isinstance(data, dict) or data.get("decision_recorded") is not True:
                return None
            if not isinstance(data.get("applies"), bool) or not isinstance(data.get("reason"), str):
                return None
            if expected_name is not None and data.get("name") != expected_name:
                return None
            return {"tool": call["name"], "applies": data["applies"], "reason": data["reason"]}
    return None


# RUNNER

def load_extraction(eis_id, extracted_dir=EXTRACTED_DIR):
    """Read one extracted result and its candidate standards.

    Returns:
        tuple: (path, record, findings list).
    """
    path = extracted_dir / f"{eis_id}.json"
    if not path.is_file():
        raise SystemExit(f"no extraction at {path}; run extract_graph.py first")
    record = json.loads(path.read_text(encoding="utf-8"))
    return path, record, record.get("standards") or []


def review_summary(findings, model):
    """Count completed decisions separately from attempts and human participation.

    Legacy Boolean verdicts without a status count as decided. An explicit
    unresolved status never counts as decided, even if it contains a Boolean.
    Empty candidate lists are complete: there was nothing to decide.
    """
    reviews = [f["review"] for f in findings if isinstance(f.get("review"), dict)]
    decided = [r for r in reviews if isinstance(r.get("applies"), bool)
               and r.get("status", "decided") == "decided"]
    seconds = sum(r.get("seconds", 0) for r in reviews)
    return {"model": model, "candidates": len(findings), "attempted": len(reviews),
            "decided": len(decided), "unresolved": len(reviews) - len(decided),
            "not_attempted": len(findings) - len(reviews),
            "human_reviewed": sum(bool(r.get("human")) for r in reviews),
            "applies": sum(r["applies"] is True for r in decided),
            "pauses": sum(r.get("pauses", 0) for r in reviews),
            "overridden": sum(bool(r.get("overridden")) for r in decided),
            "seconds": round(seconds, 1),
            "seconds_each": round(seconds / len(reviews), 1) if reviews else 0.0,
            "complete": len(decided) == len(findings)}


def main():
    """Review the standards already extracted for one procurement."""
    parser = argparse.ArgumentParser(description="Review extracted standards with a human in the loop.")
    parser.add_argument("--id", required=True, help="Procurement id under extracted/")
    parser.add_argument("--model", default=REVIEW_MODEL,
                        help="Ollama model tag, e.g. mistral-small or hf.co/user/repo")
    args = parser.parse_args()

    path, record, findings = load_extraction(args.id)
    if not findings:
        print(f"no candidate standards in {path}")
        return

    print(f"reviewing {len(findings)} candidates from {path} with {args.model}")
    agent = build_reviewer(ChatOllama(model=args.model, temperature=0, num_ctx=OLLAMA_NUM_CTX))

    # the verdict is added to each candidate; the pattern findings are left as they
    # are, so the deterministic pass stays visible next to the reviewed outcome.
    for index, finding in enumerate(findings):
        finding["review"] = review_finding(agent, finding, index)

    summary = review_summary(findings, args.model)
    reviewed = [f for f in findings if isinstance(f.get("review"), dict)]
    kept = [f for f in reviewed if f["review"].get("applies") is True
            and f["review"].get("status", "decided") == "decided"]

    record["standards_review"] = summary
    record["standards_reviewed"] = summary["complete"]
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # the run is also appended to its own file, because reviewing this procurement with
    # another model overwrites the record above but not this log, so the models can be
    # compared afterwards on the verdicts they reached.
    # the whole verdict is logged, not just the true/false, because comparing models
    # means reading the reasons they gave and seeing where I had to correct them.
    run = {**summary, "eis_id": args.id,
           "verdicts": {finding["name"]: finding["review"] for finding in reviewed}}
    with open(RUNS_PATH, "a", encoding="utf-8") as runs:
        runs.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"\n{summary['model']}: {summary['seconds']}s total, "
          f"{summary['seconds_each']}s per attempted candidate, {summary['pauses']} paused for review, "
          f"{summary['overridden']} amended by me")
    print(f"Decided {summary['decided']}, unresolved {summary['unresolved']}, "
          f"not attempted {summary['not_attempted']}, human reviewed {summary['human_reviewed']}; "
          f"kept {len(kept)} of {len(findings)} candidates")
    for finding in kept:
        print(f"  {finding['name']} ({finding['category']}, {', '.join(finding['phases'])})")
    print(f"saved {path} and appended to {RUNS_PATH}")


if __name__ == "__main__":
    main()
