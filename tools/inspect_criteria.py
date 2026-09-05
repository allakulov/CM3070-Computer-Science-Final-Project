"""Inspect how evaluation criteria would be located, without calling an LLM.

For each procurement id this loads the documents (via readers.py) and prints which
candidate keywords fired (in prose vs. table cells) and which tables the selector
would keep as scoring tables. At the end it prints a cross-procurement summary,
ranking keywords by how widely they fired.


Run:  python inspect_criteria.py 124345 125861 130001 ...
"""

import argparse
from pathlib import Path

from readers import iter_container_files, read_file


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")

# Candidate keywords to measure (English terms and strict-subset phrases removed
# after the first run). The full-run summary tells us which of these to keep.
EVAL_KEYWORDS = [
    "vērtēšanas kritērij", "izvērtēšanas kritērij", "novērtēšanas kritērij",
    "izvēles kritērij", "vērtēšanas metodik", "vērtēšanas kārtīb",
    "vērtēšanas komisij", "kritērija nosaukums", "īpatsvar", "punktu skait",
    "punktu piešķir", "maksimālais punktu", "aprēķina veids",
    "saimnieciski visizdevīgāk", "visizdevīgāko piedāvājum", "zemākā cena",
    "zemāko cenu", "zemāko piedāvāto", "viszemāk", "kvalitātes kritērij",
    "cenas kritērij",
]

# A table counts as a scoring table if at least two of its cells mention a cue.
CRITERIA_TABLE_CUES = ["kritērij", "punkt", "vērtēšan", "metodik"]


def load_documents(eis_id):
    """Read one procurement's files into (prose text, list of tables)."""
    prose, tables = [], []
    # A procurement is a folder of .zip files; iter_container_files also opens any
    # archives nested inside them and yields the real files within.
    for zip_path in sorted((DOWNLOADS_DIR / eis_id).glob("*.zip")):
        for name, data in iter_container_files(zip_path.read_bytes(), zip_path.name):
            text, file_tables = read_file(data, name)   # one parse per file
            if text:
                prose.append(text)
            tables.extend(file_tables)                  # [] when the file has no tables
    return "\n".join(prose), tables


def keyword_counts(prose, tables):
    """Return {keyword: (prose_count, table_count)} for every candidate."""
    prose = prose.lower()
    # Flatten every table cell into one lower-cased string so the tables are searchable too.
    cells = " ".join(c for t in tables for row in t["rows"] for c in row).lower()
    return {kw: (prose.count(kw), cells.count(kw)) for kw in EVAL_KEYWORDS}


def show_keywords(counts):
    """Print only the keywords that fired here, prose vs. tables, to stay scannable."""
    print(f"    {'keyword':<32} {'prose':>6} {'tables':>6}")
    for kw, (p, t) in counts.items():
        if p or t:
            print(f"    {kw:<32} {p:>6} {t:>6}")


def show_table_selection(tables):
    """Summarise the scoring-table selection, broken down by source file.

    Keep rule (same as the pipeline): a table is a scoring table if at least two
    cells contain a cue; duplicate copies (same short cells) are dropped. The
    per-source counts reveal when the one real criteria table is repeated across
    files -- which the signature dedup misses when page breaks fragment it differently.
    """
    seen = set()
    kept_by_source = {}
    dropped_dups = 0
    for t in tables:
        cells = [c for row in t["rows"] for c in row]
        hits = sum(1 for c in cells if any(cue in c.lower() for cue in CRITERIA_TABLE_CUES))
        if hits < 2:
            continue                                     # not a scoring table
        # Signature from short cells (names, weights) only; long methodology text differs between copies.
        signature = tuple(sorted(c.strip().lower() for c in cells if 0 < len(c.strip()) < 40))
        if signature in seen:
            dropped_dups += 1
            continue
        seen.add(signature)
        kept_by_source[t["source"]] = kept_by_source.get(t["source"], 0) + 1

    kept = sum(kept_by_source.values())
    print(f"  scoring tables: {kept} kept, {dropped_dups} dropped as duplicate")
    for source, n in kept_by_source.items():
        print(f"    {source}: {n}")


def show_summary(totals, n_docs):
    """Print cross-procurement keyword totals, ranked -- the block for the journal."""
    print(f"\nSUMMARY ACROSS {n_docs} PROCUREMENTS  (keyword: docs hit, prose, tables)")
    # Rank by how many procurements a keyword appeared in, then by total hits.
    ranked = sorted(totals.items(), key=lambda kv: (kv[1][0], kv[1][1] + kv[1][2]), reverse=True)
    for kw, (docs, p, t) in ranked:
        if docs:
            print(f"    {kw:<32} docs={docs:<3} prose={p:<5} tables={t}")
    dead = [kw for kw, (docs, _, _) in totals.items() if not docs]
    if dead:
        print(f"\n  never matched in any procurement (drop): {', '.join(dead)}")


def main():
    """Inspect keyword hits and table selection for each id on the command line."""
    parser = argparse.ArgumentParser(description="Inspect criteria keywords and table selection.")
    parser.add_argument("eis_ids", nargs="+", help="procurement ids under downloads/")
    args = parser.parse_args()

    totals = {kw: [0, 0, 0] for kw in EVAL_KEYWORDS}     # keyword -> [docs hit, prose hits, table hits]
    n_docs = 0
    for eis_id in args.eis_ids:
        print(f"\nPROCUREMENT {eis_id}")
        if not (DOWNLOADS_DIR / eis_id).is_dir():
            print(f"  no folder at {DOWNLOADS_DIR / eis_id}")
            continue
        n_docs += 1
        prose, tables = load_documents(eis_id)
        print(f"  loaded {len(prose):,} chars of prose, {len(tables)} tables\n")

        counts = keyword_counts(prose, tables)
        show_keywords(counts)
        print()
        show_table_selection(tables)

        # Fold this procurement's hits into the cross-id totals.
        for kw, (p, t) in counts.items():
            if p or t:
                totals[kw][0] += 1
                totals[kw][1] += p
                totals[kw][2] += t

    if n_docs:
        show_summary(totals, n_docs)


if __name__ == "__main__":
    main()