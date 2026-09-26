"""Prepare human review context without changing the agent's evidence or decisions."""
import json
import re
from pathlib import Path


def procurement_context(record):
    """Summarise saved extraction fields as context, not verified facts."""
    cpv = record.get("extracted") or {}
    lots = record.get("lots") or {}
    criteria = record.get("evaluation_criteria") or {}
    lines = [f"Procurement: {record.get('eis_id', 'unknown')}",
             f"Extracted main CPV: {cpv.get('main_cpv') or 'unknown'}",
             f"Extracted additional CPV: {', '.join(cpv.get('additional_cpv') or []) or 'none'}",
             f"Lot division: {lots.get('division', 'unknown')}; count: {lots.get('count') or 'unknown'}",
             f"Observed lot labels: {', '.join(map(str, lots.get('labels') or [])) or 'none'}",
             f"Criteria check: {record.get('criteria_check') or 'unknown'}"]
    if lots.get("warning"):
        lines.append(f"Lot warning: {lots['warning']}")
    lines.append("Extracted award criteria (context only):")
    for item in criteria.get("criteria") or []:
        lot = item.get("lot")
        lines.append(f"  Lot {lot if lot is not None else 'unspecified/shared'}: {item.get('name', '')}")
    if not criteria.get("criteria"):
        lines.append("  None extracted.")
    return "\n".join(lines)


def evidence_rows(finding):
    """Group identical saved passages without changing the original evidence."""
    rows = []
    for snippet in finding.get("evidence") or []:
        text = snippet.get("text") or ""
        phase = snippet.get("phase") or "unknown"
        existing = next((row for row in rows if row['text'] == text and row['phase'] == phase), None)
        if existing:
            existing['copies'] += 1
        else:
            rows.append({'text': text, 'phase': phase, 'copies': 1})
    return rows


def feedback_text(history):
    """Show the decisions already supplied during this candidate's review."""
    lines = []
    for index, decision in enumerate(history or [], 1):
        kind = decision.get('type')
        if kind == 'edit':
            args = (decision.get('edited_action') or {}).get('args') or {}
            detail = f"applies={args.get('applies')}; {args.get('reason', '')}"
        elif kind == 'reject':
            detail = decision.get('message', '')
        else:
            detail = 'Approved the proposal.'
        lines.append(f"{index}. {kind}: {detail}")
    return '\n'.join(lines) or 'No earlier feedback for this candidate.'


def load_support(extraction_path):
    """Read optional sibling table and OCR exports once per procurement."""
    path = Path(extraction_path)
    blocks, notes = [], []
    for folder in ('tables', 'ocr'):
        saved = path.parent.parent / folder / path.name
        try:
            documents = json.loads(saved.read_text(encoding='utf-8'))
            if not isinstance(documents, list):
                raise ValueError('expected a list of documents')
            for document in documents:
                if not isinstance(document, dict):
                    continue
                source = document.get('name') or 'Unknown source'
                if folder == 'ocr':
                    text = document.get('text') or ''
                    if text:
                        blocks.append({'source': source, 'location': 'Saved OCR text', 'text': text})
                else:
                    for index, table in enumerate(document.get('tables') or [], 1):
                        text = '\n'.join(' | '.join(str(cell or '') for cell in row) for row in table)
                        blocks.append({'source': source, 'location': f'Saved table {index}', 'text': text})
        except FileNotFoundError:
            notes.append(f'Optional file unavailable: {saved}')
        except (OSError, ValueError, TypeError) as error:
            notes.append(f'Could not read {saved}: {error}')
    return blocks, notes


def normalise(text):
    """Ignore spacing and punctuation differences introduced by table exports."""
    return re.sub(r'[^\w]+', '', text.casefold())


def supporting_passages(finding, blocks):
    """Match a substantial part of saved evidence, never a bare rule number."""
    needles = []
    for snippet in finding.get('evidence') or []:
        text = normalise(snippet.get('text') or '')
        if len(text) >= 80:
            start = max(0, (len(text) - 80) // 2)
            needles.append(text[start:start + 80])
    if not needles:
        return []
    matches = []
    seen = set()
    for block in blocks:
        key = (block['source'], block['text'])
        text = normalise(block['text'])
        if key not in seen and any(needle in text for needle in needles):
            matches.append(block)
            seen.add(key)
    return matches


def print_context(record, finding, support, history):
    """Print context and optional source material before a CLI decision."""
    print('\n' + procurement_context(record))
    print('\nReview rule: does a bidder, product or service have to meet or hold this requirement?')
    print('Phase labels and occurrence counts are extraction hints, not verified applicability.')
    print('Saved snippets can be incomplete. Supplementary material below was not added to the model prompt.')
    print('\nEarlier reviewer feedback:\n' + feedback_text(history))
    blocks, notes = support
    matches = supporting_passages(finding, blocks)
    print(f'\nOptional source matches: {len(matches)}. [e] view supporting text; [s] list source files.')
    for note in notes:
        print(note)
    return matches


def decision_source(review):
    """Identify who determined or confirmed the completed verdict."""
    if not review:
        return "Not decided"
    if review.get("status", "decided") != "decided" or not isinstance(review.get("applies"), bool):
        return "No completed decision"
    human = review.get("human") or []
    if not human:
        return "Model (automatic)"
    last = human[-1].get("type")
    if last == "edit":
        return "Human (amended)"
    if last == "approve":
        return "Model, approved by human"
    return "Model after human feedback"
