"""Summarise saved standards reviews and draw a reproducible evidence-audit sample."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def finding_row(path, record, finding, index):
    review = finding.get('review')
    attempted = isinstance(review, dict)
    review = review if attempted else {}
    applies = review.get('applies')
    decided = (isinstance(applies, bool) and review.get('status', 'decided') == 'decided')
    status = 'decided' if decided else ('unresolved' if attempted else 'not_attempted')
    human = review.get('human') or []
    seconds = review.get('seconds')
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds < 0):
        seconds = None
    passages = list(dict.fromkeys(e.get('text', '') for e in finding.get('evidence') or []))
    return {'eis_id': str(record['eis_id']), 'candidate_index': index,
            'name': finding.get('name', ''), 'category': finding.get('category', ''),
            'status': status, 'decision': ('required' if applies else 'not_required') if decided else '',
            'human_involved': bool(human), 'human_responses': len(human),
            'model': (record.get('standards_review') or {}).get('model') or 'unknown',
            'seconds': seconds, 'reason': review.get('reason', ''),
            'evidence': '\n\n'.join(passages), 'extraction_file': str(path.resolve()),
            'tables_file': str((path.parent.parent / 'tables' / path.name).resolve()),
            'ocr_file': str((path.parent.parent / 'ocr' / path.name).resolve())}


def summarise(rows):
    times = [r['seconds'] for r in rows if r['status'] != 'not_attempted' and r['seconds'] is not None]
    attempted = sum(r['status'] != 'not_attempted' for r in rows)
    decided = sum(r['status'] == 'decided' for r in rows)
    return {'candidates': len(rows), 'attempted': attempted, 'decided': decided,
            'unresolved': sum(r['status'] == 'unresolved' for r in rows),
            'not_attempted': len(rows) - attempted,
            'decision_coverage': decided / len(rows) if rows else None,
            'required': sum(r['decision'] == 'required' for r in rows),
            'not_required': sum(r['decision'] == 'not_required' for r in rows),
            'human_involved_candidates': sum(r['human_involved'] for r in rows),
            'human_responses': sum(r['human_responses'] for r in rows),
            'timed_candidates': len(times), 'attempts_missing_time': attempted - len(times),
            'agent_seconds': round(sum(times), 2) if times else None,
            'mean_seconds_per_timed_candidate': round(statistics.mean(times), 2) if times else None,
            'median_seconds_per_timed_candidate': round(statistics.median(times), 2) if times else None}


def sample_audit(rows, excluded, per_group, seed):
    groups = {}
    for row in rows:
        if row['eis_id'] in excluded or row['status'] != 'decided':
            continue
        group = ('human_involved_' if row['human_involved'] else 'automatic_') + row['decision']
        groups.setdefault(group, []).append(row)
    rng = random.Random(seed)
    sample, populations = [], {}
    for group, candidates in sorted(groups.items()):
        chosen = rng.sample(candidates, min(per_group, len(candidates)))
        populations[group] = {'population': len(candidates), 'sample': len(chosen)}
        for row in chosen:
            sample.append({**row, 'audit_group': group,
                           'group_population': len(candidates), 'group_sample': len(chosen),
                           'evidence_judgment': '', 'source_location': '', 'audit_notes': ''})
    return sample, populations


def report_table(folder, summary):
    """Write the same summary table for spreadsheets and the report."""
    measures = {
        'candidates': 'Standards candidates', 'decided': 'Decided',
        'unresolved': 'Unresolved', 'not_attempted': 'Not attempted',
        'decision_coverage': 'Decision coverage (%)',
        'required': 'Classified as required', 'not_required': 'Classified as not required',
        'human_involved_candidates': 'Candidates with human input',
        'human_responses': 'Human responses', 'timed_candidates': 'Candidates with timing',
        'attempts_missing_time': 'Attempted candidates missing timing',
        'agent_seconds': 'Total recorded agent time (seconds)',
        'mean_seconds_per_timed_candidate': 'Mean agent seconds per timed candidate',
        'median_seconds_per_timed_candidate': 'Median agent seconds per timed candidate',
    }
    rows = [{'Measure': 'Procurements', 'All procurements': summary['procurements'],
             'Audit scope': summary['procurements'] - len(summary['audit_excluded_ids'])}]
    for key, label in measures.items():
        row = {'Measure': label}
        for column, scope in [('All procurements', 'all_findings'), ('Audit scope', 'audit_scope')]:
            value = summary[scope][key]
            row[column] = round(value * 100, 1) if key == 'decision_coverage' and value is not None else value
        rows.append(row)
    write_csv(folder / 'report_table.csv', rows, list(rows[0]))
    lines = ['| Measure | All procurements | Audit scope |', '| --- | ---: | ---: |']
    for row in rows:
        cells = ['Not available' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)
                 for v in row.values()]
        lines.append('| ' + ' | '.join(cells) + ' |')
    excluded = ', '.join(summary['audit_excluded_ids']) or 'None'
    lines += ['', f'Audit exclusions: {excluded}. Coverage and runtime use all selected files in the first column.',
              'Agent time includes retries and excludes human waiting. Decision coverage is decided / candidates.',
              'Audit scope describes the eligible population, not the sampled cases. Required decisions are not independently verified recovery.']
    (folder / 'report_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extracted-dir', type=Path, required=True)
    parser.add_argument('--ids-file', type=Path, help='Optional whitespace-separated procurement IDs')
    parser.add_argument('--exclude-from-audit', nargs='*', default=[])
    parser.add_argument('--per-group', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', type=Path, default=Path('standards_analysis'))
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error('--per-group must be positive')
    if args.ids_file:
        ids = args.ids_file.read_text().split()
        if len(ids) != len(set(ids)) or not all(i.isdigit() for i in ids):
            parser.error('IDs must be unique numbers')
        paths = [args.extracted_dir / f'{i}.json' for i in sorted(ids)]
    else:
        paths = sorted(p for p in args.extracted_dir.glob('*.json') if p.stem.isdigit())
    if not paths:
        parser.error('No extraction files selected')
    rows, procurements, seen = [], [], set()
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        eis_id = str(record.get('eis_id'))
        if eis_id != path.stem or eis_id in seen:
            raise ValueError(f'Missing, mismatched or duplicate procurement ID: {path}')
        seen.add(eis_id)
        findings = record.get('standards') or []
        current = [finding_row(path, record, f, i) for i, f in enumerate(findings, 1)]
        rows.extend(current)
        procurements.append({'eis_id': eis_id, **summarise(current)})
    excluded = set(args.exclude_from_audit)
    if excluded - seen:
        parser.error('Audit exclusion IDs not in the selected files: ' + ', '.join(sorted(excluded - seen)))
    sample, populations = sample_audit(rows, excluded, args.per_group, args.seed)
    summary = {'procurements': len(paths), 'all_findings': summarise(rows),
               'by_model': {model: summarise([r for r in rows if r['model'] == model])
                            for model in sorted({r['model'] for r in rows})},
               'audit_excluded_ids': sorted(excluded),
               'excluded_findings': summarise([r for r in rows if r['eis_id'] in excluded]),
               'audit_scope': summarise([r for r in rows if r['eis_id'] not in excluded]),
               'audit_seed': args.seed, 'audit_groups': populations,
               'runtime_definition': 'Saved agent-call time including retries; excludes human waiting in the supplied review modules.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / 'audit_sample.csv'
    if sample_path.exists():
        parser.error('audit_sample.csv already exists. Choose a new output directory to preserve annotations.')
    columns = list(rows[0]) if rows else ['eis_id', 'candidate_index', 'name', 'status']
    write_csv(args.output_dir / 'findings.csv', rows, columns)
    write_csv(args.output_dir / 'procurements.csv', procurements, list(procurements[0]))
    write_csv(sample_path, sample, ['audit_group', 'group_population', 'group_sample'] + columns +
              ['evidence_judgment', 'source_location', 'audit_notes'])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_table(args.output_dir, summary)
    print(f"Saved report_table.csv, report_table.md and detailed results to {args.output_dir}")
    print(f"Audit sample: {len(sample)} candidates. Seed: {args.seed}")


if __name__ == '__main__':
    main()
