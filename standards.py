"""Find the standards and certificates required in a procurement's documents.

A deterministic pass over the document text: regexes for coded standards (ISO, EN,
LVS, IEC) and for national and EU legal references, plus a keyword search for named
certification and ecolabel schemes. No model is involved, so a whole corpus can be
swept quickly, and extract_graph.py calls find_standards directly as one branch.

The scheme names and cues live in standards_catalog.json rather than in this file,
so the module can be taught a new scheme by editing data. On top of that, schemes a
notice defines for itself are picked up as it is read: Latvian notices introduce
their own abbreviations, as in "bioloģiskās lauksaimniecības (turpmāk - BL)", so a
national scheme is found even when the catalog has never seen it.

Repeats of the same standard are merged into one finding that records every tender
phase it appears in (selection and specification are requirements the bidder must
meet, award means it earns points) and keeps a few passages of evidence. A green
flag marks the environmental schemes and standards, which is the raw material for a
later green-procurement label.

Run:
    python standards.py --id 123450
    python standards.py --file some_text.txt
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path


# CONFIGURATION

DOWNLOADS_DIR = Path("downloads")                                # input: downloads/{eis_id}/*.zip
CATALOG_PATH = Path(__file__).with_name("standards_catalog.json")  # the editable data file
EVIDENCE_CHARS = 150                                             # context kept each side of a hit
MAX_EVIDENCE = 4                                                 # passages kept per standard


# PATTERNS
#
# These describe the shape of a reference rather than any particular one, so they
# stay in code while the named schemes and cues live in the catalog.

# a standard is a body, sometimes several, then its number: ISO 8601,
# ISO 37001:2025, ISO/IEC 17025:2017, LVS EN 1090-2.
BODY = r"(?:LVS|EN|ISO/IEC|ISO|IEC)"
NUMBER = r"\d{3,5}(?:-\d{1,3})?(?::\s?\d{4})?"
STANDARD_PATTERN = re.compile(rf"\b{BODY}(?:\s+{BODY})*\s*{NUMBER}\b", re.IGNORECASE)

# national rules, e.g. "Ministru kabineta 2013. gada 8. oktobra noteikumi Nr. 1041".
NATIONAL_PATTERN = re.compile(
    r"Ministru kabineta[^.\n]{0,80}?noteikum\w*\s*Nr\.?\s?(\d+)", re.IGNORECASE)

# an EU act is known by its number, with the union named before it, after it, or
# not at all: "regulu (EK) Nr. 66/2010", "Direktīvu 95/46/EK", "regulai 2016/7".
EU_PATTERNS = [
    re.compile(r"\((?:ES|EK|EEK)\)\s*(?:Nr\.?\s*)?(\d{1,4}/\d{1,4})"),
    re.compile(r"(\d{1,4}/\d{1,4})/E[KS]\b"),
    re.compile(r"regul\w+\s+(\d{1,4}/\d{1,4})", re.IGNORECASE),
]

# an abbreviation a notice defines for itself: "... shēmas (turpmāk - NPKS)".
# only the bracket is matched; what it stands for is read from the text before it.
DEFINED_PATTERN = re.compile(r"\(\s*turpm[āa]k[^)]{0,20}?([A-ZĀČĒĢĪĶĻŅŠŪŽ]{2,6})\s*\)")
MEANING_CHARS = 70                                               # text read back for the meaning


# CATALOG

CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
SCHEMES = CATALOG["schemes"]
GREEN_CODES = CATALOG["green_codes"]
GREEN_REGULATIONS = CATALOG["green_regulations"]
GREEN_CUES = CATALOG["green_cues"]
SCHEME_CUES = CATALOG["scheme_cues"]
SECTION_MARKERS = CATALOG["section_markers"]


# HELPERS

def fold(text):
    """Lower-case the text and strip accents, one character in, one character out.

    OCR and PDF extraction often drop Latvian diacritics, so keywords are matched
    against this plain form. NFD splits an accented letter into its plain letter plus
    a separate accent mark, so taking the first character gives the plain letter.
    Keeping it one character to one character means a position in the folded text is
    the same position in the original, so evidence can be sliced from the original.
    """
    return "".join(unicodedata.normalize("NFD", char)[0] for char in text).lower()


def find_keyword(folded, keyword):
    """Return the (start, end) of every folded keyword hit that starts a word.

    Only the left side is checked, so a stem matches its inflections. Any run of
    whitespace matches between words, because extracted text wraps across lines.
    """
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in keyword.split())
    return [(match.start(), match.end()) for match in re.finditer(pattern, folded)]


def evidence_for(text, start, end):
    """Return a whitespace-collapsed window of text around a match."""
    left = max(0, start - EVIDENCE_CHARS)
    right = min(len(text), end + EVIDENCE_CHARS)
    return " ".join(text[left:right].split())


def phase_at(position, headers):
    """Return the phase of the nearest section heading above a position."""
    phase = "unknown"
    for header_position, header_phase in headers:
        if header_position > position:
            break
        phase = header_phase
    return phase


def sample_evidence(occurrences):
    """Keep a few passages, spread across the places the standard is named.

    A standard can be named dozens of times in near-identical wording, so keeping
    every passage would swamp the prompt. Taking them at an even spacing keeps early
    and late mentions alike, which matters because the wording that decides whether a
    standard applies often comes well after the first passing mention.
    """
    if MAX_EVIDENCE <= 0:
        return []
    if len(occurrences) > MAX_EVIDENCE:
        if MAX_EVIDENCE == 1:
            occurrences = occurrences[:1]  # one slot cannot retain both endpoints
        else:
            last = len(occurrences) - 1
            occurrences = [occurrences[i * last // (MAX_EVIDENCE - 1)]
                           for i in range(MAX_EVIDENCE)]
    return [{"phase": phase, "text": text} for phase, text in occurrences]


# CORE

def find_standards(text, tables=None):
    """Find the standards and certificates named in one procurement.

    Table cells are appended to the text so a standard named only inside a technical
    specification table is still caught.

    Returns:
        list: One finding per standard, with its category, phases, count and evidence.
    """
    if tables:
        cells = [str(cell) for table in tables for row in table.get("rows", []) for cell in row if cell]
        text = text + "\n" + " ".join(cells)

    folded = fold(text)
    headers = sorted((position, phase)
                     for phase, cues in SECTION_MARKERS.items()
                     for cue in cues
                     for position, _ in find_keyword(folded, cue))
    found = {}

    def add(name, category, green, start, end):
        """Record one occurrence, merging repeats of the same standard."""
        finding = found.setdefault(name, {"name": name, "category": category,
                                          "green": green, "count": 0,
                                          "phases": [], "occurrences": []})
        phase = phase_at(start, headers)
        finding["count"] += 1
        if phase not in finding["phases"]:
            finding["phases"].append(phase)
        finding["occurrences"].append((phase, evidence_for(text, start, end)))

    for match in STANDARD_PATTERN.finditer(text):
        name = " ".join(match.group().split()).upper()
        green = any(code in name.split(":")[0] for code in GREEN_CODES)
        add(name, "standard", green, match.start(), match.end())

    for match in NATIONAL_PATTERN.finditer(text):
        add(f"Ministru kabineta noteikumi Nr. {match.group(1)}", "regulation", False,
            match.start(), match.end())

    for pattern in EU_PATTERNS:
        for match in pattern.finditer(text):
            number = match.group(1)
            add(f"EU regulation {number}", "regulation", number in GREEN_REGULATIONS,
                match.start(), match.end())

    for scheme, entry in SCHEMES.items():
        for alias in entry["aliases"]:
            for start, end in find_keyword(folded, alias):
                add(scheme, "scheme", entry["green"], start, end)

    # schemes the notice defines for itself, kept only when the wording names a scheme
    for match in DEFINED_PATTERN.finditer(text):
        meaning = fold(text[max(0, match.start() - MEANING_CHARS):match.start()])
        if not any(cue in meaning for cue in SCHEME_CUES):
            continue
        acronym = match.group(1)
        green = any(cue in meaning for cue in GREEN_CUES)
        for use in re.finditer(r"\b" + re.escape(acronym) + r"\b", text):
            add(acronym, "defined_scheme", green, use.start(), use.end())

    for finding in found.values():
        finding["evidence"] = sample_evidence(finding.pop("occurrences"))
    return sorted(found.values(), key=lambda finding: (finding["category"], finding["name"]))


# RUNNER

def read_procurement(eis_id):
    """Read one procurement's documents into text and tables, for a standalone sweep."""
    from readers import iter_container_files, read_file

    texts, tables = [], []
    for zip_path in sorted((DOWNLOADS_DIR / eis_id).glob("*.zip")):
        zip_bytes = zip_path.read_bytes()
        for name, data in iter_container_files(zip_bytes, zip_path.name):
            text, file_tables = read_file(data, name)    # one parse per file
            if text:
                texts.append(text)
            tables.extend(file_tables)
    return "\n\n".join(texts), tables


def print_report(standards):
    """Print the findings in a readable form."""
    if not standards:
        print("no standards or certificates found")
        return
    print(f"found {len(standards)} standards and certificates:")
    for finding in standards:
        green = "  [green]" if finding["green"] else ""
        phases = ", ".join(finding["phases"])
        print(f"  [{finding['category']}] {finding['name']} "
              f"(phases: {phases}, seen {finding['count']}x){green}")
        for passage in finding["evidence"]:
            print(f"      ({passage['phase']}) {passage['text']}")


def main():
    """Scan one procurement or a text file for standards and certificates."""
    parser = argparse.ArgumentParser(description="Find required standards and certificates.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--id", help="Procurement id under downloads/")
    source.add_argument("--file", help="A plain text file to scan")
    parser.add_argument("--json", action="store_true", help="Print the raw result as JSON")
    args = parser.parse_args()

    if args.file:
        text, tables = Path(args.file).read_text(encoding="utf-8"), None
    else:
        text, tables = read_procurement(args.id)

    standards = find_standards(text, tables)
    if args.json:
        print(json.dumps(standards, indent=2, ensure_ascii=False))
    else:
        print_report(standards)


if __name__ == "__main__":
    main()
