"""Compare three lot-search patterns in procurement documents.

Place beside readers.py and run:
    python compare_lot_patterns.py --downloads downloads

Outputs: lot_matches.csv and lot_matches_summary.csv.
Read each document once, test all three patterns, then inspect their examples.
No reference JSON, previous audit script, cache or model calls are required.

The patterns progress from a broad word to a procurement-specific declaration.
The last is the selected rule from earlier exploration. This compact experiment
lets you reproduce and inspect the comparison; it does not select a winner by
match count. A useful match establishes division/non-division or lot structure.
Legal provisions, document sections and proposed changes can be irrelevant.
Mark useful and notes in the matches CSV after reading the context.

Counts are matching lines, not distinct facts. Prose and tables may repeat
content. Missing matches do not establish absence of lots. Reader failures are
printed. Searches are line-based, so wrapped declarations may be missed.
"""
import argparse
import csv
from pathlib import Path
import re

from readers import iter_container_files, read_file

# Three alternatives, from broad to specific.
PATTERNS = {
    '1_parts_word': r"\bdaļ\w*\b",
    '2_division_phrase': r"\b(?:sa|ne)?dalīt\w*[^\n:;.!?]{0,80}\bdaļ\w*",
    '3_procurement_declaration': (
        r"\biepirkuma\s+priekšmets\s+"
        r"(?:(?:ir|tiek|nav)\s+)?"
        r"(?:sa|ne)?dalīt\w*"
        r"[^\n:;.!?]{0,80}\bdaļ\w*"
    ),
}
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in PATTERNS.items()}


def find_matches(text, pattern):
    """Keep a matching line, one preceding line and 15 following lines."""
    lines = text.splitlines()
    matches = []
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            start = max(0, index - 1)
            context = '\n'.join(lines[start:index + 16])
            matches.append((index + 1, match.group(), context))
    return matches


def search_document(procurement_id, name, data, writer, counts):
    """Search both ordinary text and tables returned by the existing reader."""
    info = {}
    text, tables = read_file(data, name, info)
    if not text and not tables:
        print(f'  Unread: {name}: {info.get("reason", "no content")}')
        return
    sections = [('text', text)]
    for number, table in enumerate(tables, 1):
        rows = []
        for row in table['rows']:
            rows.append('\t'.join('' if cell is None else str(cell) for cell in row))
        sections.append((f'table {number}', '\n'.join(rows)))
    for section, content in sections:
        for label, pattern in PATTERNS.items():
            for line, match, context in find_matches(content, pattern):
                writer.writerow([procurement_id, label, name, section, line,
                                 match, context, '', ''])
                counts[label] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path('downloads'))
    parser.add_argument('--ids', nargs='+', help='Optional procurement IDs')
    parser.add_argument('--out', type=Path, default=Path('lot_matches.csv'))
    args = parser.parse_args()
    if not args.downloads.is_dir():
        parser.error(f'Downloads folder not found: {args.downloads}')
    folders = sorted(p for p in args.downloads.iterdir() if p.is_dir() and p.name.isdigit())
    if args.ids:
        folders = [args.downloads / procurement_id for procurement_id in args.ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    with args.out.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['id', 'pattern', 'source', 'section', 'line', 'match', 'context', 'useful', 'notes'])
        for folder in folders:
            if not folder.is_dir():
                print(f'Skipped missing folder: {folder}')
                continue
            print(f'Reading {folder.name}...', flush=True)
            counts = {name: 0 for name in PATTERNS}
            for path in sorted(folder.rglob('*')):
                if not path.is_file():
                    continue
                name = str(path.relative_to(folder))
                try:
                    data = path.read_bytes()
                    if path.suffix.lower() in {'.zip', '.edoc'}:
                        documents = iter_container_files(data, name)
                    else:
                        documents = [(name, data)]
                    for source, payload in documents:
                        try:
                            search_document(folder.name, source, payload, writer, counts)
                        except Exception as error:
                            print(f'  Cannot read {source}: {error}')
                except Exception as error:
                    print(f'  Cannot open {name}: {error}')
            summary.append([folder.name, *counts.values()])
            print(f'{folder.name}: {counts}')
    summary_path = args.out.with_name(args.out.stem + '_summary.csv')
    with summary_path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['id', *PATTERNS])
        writer.writerows(summary)
    print(f'Saved {args.out} and {summary_path}. Review the passages, not just counts.')


if __name__ == '__main__':
    main()
