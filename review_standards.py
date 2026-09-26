"""Review saved standards using evidence, tool decisions and human input.

Run: python review_standards.py --id 123450 --extracted-dir extracted
Human-in-the-loop pattern: https://docs.langchain.com/oss/python/langchain/human-in-the-loop
"""
import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

from review_context import load_support, print_context, evidence_rows
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

EXTRACTED_DIR = Path("extracted")
RUNS_PATH = Path("standards_review_runs.jsonl")
REVIEW_MODEL = "gemma4:e4b"
OLLAMA_NUM_CTX = 4096



REVIEW_PROMPT = """Review this candidate using ONLY the supplied evidence.
Do not use your knowledge of the standard, certificate or law to fill gaps.

Decide whether the evidence requires a bidder, product or service to meet
or hold this standard, certificate or scheme.

Choose exactly one tool:
- record_decision: only when the evidence clearly supports a decision.
  Set applies=true for an explicit requirement to meet or hold it.
  Set applies=false for an explicit exclusion, an example, unrelated text,
  or a general legal obligation without such a requirement.
- request_review: when evidence is missing, too short, ambiguous or conflicting,
  or you cannot tell whether the rule above is satisfied.
  Its applies value is a provisional suggestion for the human, not a final decision.

Absence of a clear requirement in a short excerpt is not evidence of non-applicability.
A familiar name, category, phase or mention count is not enough to decide.
Give one short reason tied to the supplied wording. When requesting review,
state what is missing or unclear. Do not invent supporting facts.

CANDIDATE: {name}
CATEGORY: {category}
INFERRED PHASES: {phases}; MENTIONS: {count}
EVIDENCE:
{evidence}
"""

@tool(return_direct=True, response_format="content_and_artifact")
def record_decision(name: str, applies: bool, reason: str) -> tuple[str, dict]:
    """Record a clear decision about a candidate standard."""
    receipt = {"decision_recorded": True, "name": name, "applies": applies, "reason": reason}
    return json.dumps(receipt), receipt


@tool(return_direct=True, response_format="content_and_artifact")
def request_review(name: str, applies: bool, reason: str) -> tuple[str, dict]:
    """Ask a person to decide about a candidate standard."""
    receipt = {"decision_recorded": True, "name": name, "applies": applies, "reason": reason}
    return json.dumps(receipt), receipt


def build_reviewer(model=None):
    """Build the review agent with the human-in-the-loop middleware."""
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


def format_evidence(finding):
    """Lay out every evidence passage for one candidate, labelled by phase."""
    return "\n".join(f"- ({snippet['phase']}) {snippet['text']}"
                     for snippet in finding["evidence"])


def load_extraction(eis_id, extracted_dir=EXTRACTED_DIR):
    """Read one extracted result and its candidate standards."""
    path = extracted_dir / f"{eis_id}.json"
    if not path.is_file():
        raise SystemExit(f"no extraction at {path}; run extract_graph.py first")
    record = json.loads(path.read_text(encoding="utf-8"))
    return path, record, record.get("standards") or []


def review_summary(findings, model):
    """Count completed decisions separately from attempts and human participation."""
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



def last_tool_call(result, expected_name=None):
    """Read a completed decision, matching its tool call and candidate."""
    messages = result.value["messages"]
    for index in range(len(messages) - 1, -1, -1):
        for call in reversed(getattr(messages[index], "tool_calls", []) or []):
            if call["name"] not in {"record_decision", "request_review"}:
                continue
            replies = [m for m in messages[index + 1:]
                       if getattr(m, "type", None) == "tool"
                       and getattr(m, "tool_call_id", None) == call["id"]]
            if not replies or getattr(replies[-1], "status", "success") == "error":
                return None
            data = getattr(replies[-1], "artifact", None)
            if not isinstance(data, dict) or data.get("decision_recorded") is not True:
                return None
            if expected_name is not None and data.get("name") != expected_name:
                return None
            if not isinstance(data.get("applies"), bool) or not isinstance(data.get("reason"), str):
                return None
            return {"tool": call["name"], "applies": data["applies"], "reason": data["reason"]}
    return None


class ReviewSession:
    """Review one candidate; a pending request must be answered before completion."""

    def __init__(self, agent, finding):
        self.agent, self.finding = agent, finding
        self.config = {"configurable": {"thread_id": str(uuid4())}}
        self.human, self.pending, self.decisions = [], [], []
        self.seconds, self.pauses = 0.0, 0
        self.proposed, self.verdict = None, None
        self.manual = False

    @property
    def request(self):
        return self.pending[0] if self.pending else None

    def start(self):
        prompt = REVIEW_PROMPT.format(
            name=self.finding["name"], category=self.finding["category"],
            phases=", ".join(self.finding["phases"]), count=self.finding["count"],
            evidence=format_evidence(self.finding))
        self.invoke({"messages": [{"role": "user", "content": prompt}]})

    def invoke(self, message):
        started = time.perf_counter()
        try:
            result = self.agent.invoke(message, config=self.config, version="v2")
        except Exception as error:
            self.ask_directly(f"The model call failed ({type(error).__name__}). Please review the evidence.")
            return
        finally:
            self.seconds += time.perf_counter() - started
        if result.interrupts:
            self.pending = [request for interrupt in result.interrupts
                            for request in interrupt.value["action_requests"]]
            self.manual = False
            self.pauses += 1
            if self.pending:
                if self.proposed is None:
                    self.proposed = self.request.get("args", {}).get("applies")
                return
        verdict = last_tool_call(result, self.finding["name"])
        if verdict is None:
            self.ask_directly("The model returned no completed decision. Please review the evidence.")
        else:
            # An amendment supplies the final decision and reason.
            if self.human and self.human[-1]["type"] == "edit":
                args = self.human[-1]["edited_action"]["args"]
                verdict.update(applies=args["applies"], reason=args["reason"])
            self.finish(verdict)

    def ask_directly(self, reason):
        self.manual = True
        self.pending = [{"name": "request_review", "args": {
            "name": self.finding["name"], "applies": None, "reason": reason}}]

    def answer(self, decision):
        if not self.pending:
            raise ValueError("There is no pending human decision.")
        kind = decision.get("type")
        if kind not in {"approve", "edit", "reject"}:
            raise ValueError("Choose approve, edit or reject.")
        if self.manual and kind == "approve":
            raise ValueError("Choose required or not required; there is no model decision to approve.")
        if kind == "edit":
            args = decision["edited_action"]["args"]
            if (args.get("name") != self.finding["name"]
                    or not isinstance(args.get("applies"), bool)
                    or not isinstance(args.get("reason"), str) or not args["reason"].strip()):
                raise ValueError("A human decision needs the candidate name, a Boolean value and a reason.")
        self.human.append(decision)
        if self.manual:
            self.pending = []
            if kind == "edit":
                self.finish({"tool": "human_decision", "applies": args["applies"], "reason": args["reason"]})
            else:
                self.invoke({"messages": [{"role": "user", "content": decision["message"]}]})
            return
        self.decisions.append(decision)
        self.pending.pop(0)
        if not self.pending:
            decisions, self.decisions = self.decisions, []
            self.invoke(Command(resume={"decisions": decisions}))

    def finish(self, verdict):
        proposed = self.proposed if self.proposed is not None else verdict["applies"]
        self.verdict = {**verdict, "status": "decided", "human": self.human,
                        "seconds": round(self.seconds, 1), "pauses": self.pauses,
                        "proposed": proposed, "overridden": proposed != verdict["applies"],
                        "decision_source": "human" if self.human and self.human[-1]["type"] in {"edit", "approve"} else "model"}
        self.pending = []


def make_decision(request, applies, reason):
    """Create an amendment or a direct human decision."""
    return {"type": "edit", "edited_action": {"name": request["name"],
            "args": {**request["args"], "applies": applies, "reason": reason}}}


def ask_person(finding, request, record=None, support=None, history=None):
    """Show the evidence and collect approval, a decision or more information."""
    matches = print_context(record or {}, finding, support or ([], []), history)
    for snippet in evidence_rows(finding):
        print(f"  {snippet['phase']}: {snippet['text']}")
    args = request["args"]
    print(f"\n{finding['name']}: {args['reason']}")
    if isinstance(args.get("applies"), bool):
        print("Model decision:", "Required" if args["applies"] else "Not required")
    while True:
        answer = input("[a] approve, [m] decide, [i] send information, [e] more evidence, [s] sources: ").lower().strip()
        if answer == "a" and isinstance(args.get("applies"), bool):
            return {"type": "approve"}
        if answer == "m":
            value = input("Required? [y/n]: ").lower().strip()
            reason = input("Reason: ").strip()
            if value in {"y", "n"} and reason:
                return make_decision(request, value == "y", reason)
        elif answer == "i":
            note = input("Evidence or clarification for the model: ").strip()
            if note:
                return {"type": "reject", "message": note}
        elif answer == "e":
            for block in matches:
                print(f"{block['source']} | {block['location']}\n{block['text']}")
            if not matches:
                print("No matching saved table or OCR text.")
        elif answer == "s":
            print("\n".join((record or {}).get("source_files") or []))
        else:
            print("Choose a listed action. Approval requires an existing model decision.")


def review_finding(agent, finding, index=None, record=None, support=None):
    """Review one candidate until a model or human decision is recorded."""
    session = ReviewSession(agent, finding)
    session.start()
    while session.request:
        session.answer(ask_person(finding, session.request, record, support, session.human))
    return session.verdict


def save_review(path, record, findings, model_tag):
    """Save decisions and append the run's measurements to the review log."""
    summary = review_summary(findings, model_tag)
    record.update(standards=findings, standards_review=summary, standards_reviewed=summary["complete"])
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    run = {**summary, "eis_id": record.get("eis_id"),
           "verdicts": {f["name"]: f["review"] for f in findings if f.get("review")}}
    with RUNS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--model", default=REVIEW_MODEL)
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    args = parser.parse_args()
    path, record, findings = load_extraction(args.id, args.extracted_dir)
    agent = build_reviewer(ChatOllama(model=args.model, temperature=0, num_ctx=OLLAMA_NUM_CTX))
    support = load_support(path)
    for finding in findings:
        finding["review"] = review_finding(agent, finding, record=record, support=support)
    summary = save_review(path, record, findings, args.model)
    print(f"Decided {summary['decided']}; required {summary['applies']}; "
          f"human reviewed {summary['human_reviewed']}; agent time {summary['seconds']} seconds.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
