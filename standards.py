"""Find the standards and certificates required in a procurement's documents.

A deterministic pass over the document text: regexes for coded standards (ISO, EN,
LVS, IEC) and for national and EU legal references, plus a keyword search for named
certification and ecolabel schemes. No model is involved, so a whole corpus can be
swept quickly, and extract_graph.py calls find_standards directly as one branch.

The scheme names, cues and keywords live in standards_catalog.json rather than in
this file, so the module can be taught a new scheme by editing data. On top of that,
schemes a notice defines for itself are discovered as it is read: Latvian notices
introduce their own abbreviations, as in "bioloģiskās lauksaimniecības (turpmāk -
BL)", so a national scheme is picked up even when the catalog has never seen it.

Each finding carries the tender phase it sits in (selection and specification are
requirements the bidder must meet, award means the standard earns points) and a
green flag marking the environmental schemes and standards, which is the raw
material for a later green-procurement label.

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
EVIDENCE_CHARS = 150                                              # context kept each side of a hit


# PATTERNS
#
# These describe the shape of a reference rather than any particular one, so they
# stay in code while the named schemes and cues live in the catalog.

# a standard is one or more body tokens (LVS EN ISO ...) then its number, with an
# optional part number and an optional year: ISO 8601, ISO 37001:2025,
# ISO/IEC 17025:2017, LVS EN 1090-2. ISO/IEC is listed before ISO so it wins.
STANDARD_PATTERN = re.compile(
    r"\b(?:LVS|EN|ISO/IEC|ISO|IEC)(?:\s+(?:LVS|EN|ISO/IEC|ISO|IEC))*"
    r"\s*\d{3,5}(?:-\d{1,3})?(?:\s?:\s?\d{4})?\b",
    re.IGNORECASE,
)

# national rules, e.g. "Ministru kabineta 2013. gada 8. oktobra noteikumi Nr. 1041".
NATIONAL_REGULATION_PATTERN = re.compile(
    r"Ministru kabineta[^.\n]{0,80}?noteikum\w*\s*Nr\.?\s?(\d+)", re.IGNORECASE)

# EU rules, e.g. "regulu (EK) Nr.66/2010", "Īstenošanas regulai 2016/7".
EU_REGULATION_PATTERN = re.compile(
    r"(regul\w+|direkt[īi]v\w+)[^\n]{0,40}?(\d{1,4}/\d{1,4})", re.IGNORECASE)

# an abbreviation a notice defines for itself: "... shēmas (turpmāk - NPKS)".
# the expansion may wrap across lines, so newlines are allowed inside it.
DEFINED_SCHEME_PATTERN = re.compile(
    r"([^.(]{5,70}?)\s*\(\s*turpm[āa]k[^)]{0,30}?[-–—]\s*([A-ZĀČĒĢĪĶĻŅŠŪŽ]{2,6})\s*\)")


# CATALOG

def load_catalog(path=CATALOG_PATH):
    """Load the scheme names, cues and keywords from the data file."""
    return json.loads(path.read_text(encoding="utf-8"))


CATALOG = load_catalog()
SCHEMES = CATALOG["schemes"]
GREEN_CODES = CATALOG["green_codes"]
GREEN_REGULATIONS = CATALOG["green_regulations"]
SCHEME_CUES = CATALOG["scheme_cues"]
GREEN_KEYWORDS = CATALOG["green_keywords"]
SECTION_MARKERS = CATALOG["section_markers"]


# HELPERS

def fold(text):
    """Lower-case the text and strip accents, one character in, one character out.

    OCR and PDF extraction often drop Latvian diacritics, so keywords are matched
    against this plain form. NFD splits an accented letter into its plain letter
    plus a separate accent mark, so taking the first character of that split gives
    the plain letter. Keeping it one character to one character means a position in
    the folded text is the same position in the original, so evidence can still be
    sliced from the original text.
    """
    return "".join(unicodedata.normalize("NFD", char)[0] for char in text).lower()


def find_keyword(folded, keyword):
    """Return the (start, end) of every folded keyword hit that starts a word.

    Only the left side is checked, so a stem matches its inflections:
    "energoefektiv" hits "energoefektivitātes". Any run of whitespace matches
    between words, because extracted text wraps a phrase across lines.
    """
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in keyword.split())
    return [(match.start(), match.end()) for match in re.finditer(pattern, folded)]


def evidence_for(text, start, end):
    """Return a whitespace-collapsed window of text around a match."""
    left = max(0, start - EVIDENCE_CHARS)
    right = min(len(text), end + EVIDENCE_CHARS)
    return " ".join(text[left:right].split())


def section_headers(folded):
    """Return sorted (position, phase) pairs for every section heading found."""
    headers = []
    for phase, cues in SECTION_MARKERS.items():
        for cue in cues:
            headers.extend((pos, phase) for pos, _ in find_keyword(folded, cue))
    headers.sort()
    return headers


def phase_at(position, headers):
    """Return the phase of the nearest heading above a position."""
    phase = "unknown"
    for header_pos, header_phase in headers:
        if header_pos > position:
            break
        phase = header_phase
    return phase


def defined_schemes(text, folded):
    """Find abbreviations the notice defines for itself that name a scheme.

    A notice introduces its own terms, as in "nacionālās pārtikas kvalitātes shēmas
    (turpmāk - NPKS)". A definition is treated as a scheme when its expansion holds
    a cue from the catalog, which keeps genuine schemes and drops the many
    procedural abbreviations a notice also defines.
    """
    schemes = {}
    for match in DEFINED_SCHEME_PATTERN.finditer(text):
        expansion = " ".join(match.group(1).split())
        acronym = match.group(2)
        folded_expansion = fold(expansion)
        if not any(cue in folded_expansion for cue in SCHEME_CUES):
            continue
        if acronym not in schemes:
            green = any(word in folded_expansion for word in GREEN_KEYWORDS)
            schemes[acronym] = (expansion[-60:].strip(" ,;-"), green)
    return schemes


# CORE

def find_standards(text, tables=None):
    """Find the standards, certificates and green language in one procurement.

    Table cells are appended to the text so a standard named only inside a technical
    specification table is still caught. Repeated findings are collapsed by name and
    phase, keeping a count and the first piece of evidence.
    """
    if tables:
        cells = [str(cell) for table in tables for row in table.get("rows", []) for cell in row if cell]
        text = text + "\n" + " ".join(cells)

    folded = fold(text)
    headers = section_headers(folded)
    found = {}

    def add(name, category, green, start, end):
        """Record one finding, or count it again if already seen in this phase."""
        key = (name, phase_at(start, headers))
        if key in found:
            found[key]["count"] += 1
        else:
            found[key] = {"name": name, "category": category, "phase": key[1],
                          "green": green, "count": 1,
                          "evidence": evidence_for(text, start, end)}

    for match in STANDARD_PATTERN.finditer(text):
        name = " ".join(match.group().split()).upper()
        base = name.split(":")[0].strip()
        add(name, "standard", any(code in base for code in GREEN_CODES), match.start(), match.end())

    for match in NATIONAL_REGULATION_PATTERN.finditer(text):
        add(f"Ministru kabineta noteikumi Nr. {match.group(1)}", "regulation", False,
            match.start(), match.end())

    for match in EU_REGULATION_PATTERN.finditer(text):
        number = match.group(2)
        kind = "regulation" if match.group(1).lower().startswith("regul") else "directive"
        add(f"EU {kind} {number}", "regulation", number in GREEN_REGULATIONS,
            match.start(), match.end())

    for scheme, entry in SCHEMES.items():
        for alias in entry["aliases"]:
            for start, end in find_keyword(folded, alias):
                add(scheme, "scheme", entry["green"], start, end)

    # schemes the notice defines for itself, found by reading its own abbreviations
    for acronym, (expansion, green) in defined_schemes(text, folded).items():
        for match in re.finditer(r"\b" + re.escape(acronym) + r"\b", text):
            add(f"{acronym} ({expansion})", "defined_scheme", green, match.start(), match.end())

    green_signals = []
    for keyword in GREEN_KEYWORDS:
        spots = find_keyword(folded, keyword)
        if spots:
            green_signals.append({"keyword": keyword, "count": len(spots),
                                  "evidence": evidence_for(text, *spots[0])})

    standards = sorted(found.values(), key=lambda f: (f["category"], f["name"]))
    return {"standards": standards, "green_signals": green_signals}


# RUNNER

def read_procurement(eis_id):
    """Read one procurement's documents into text and tables, for a standalone sweep."""
    from readers import iter_container_files, read_text, extract_tables

    texts, tables = [], []
    for zip_path in sorted((DOWNLOADS_DIR / eis_id).glob("*.zip")):
        zip_bytes = zip_path.read_bytes()
        for name, data in iter_container_files(zip_bytes, zip_path.name):
            text = read_text(data, name)
            if text:
                texts.append(text)
            tables.extend(extract_tables(data, name))
    return "\n\n".join(texts), tables


def print_report(result):
    """Print the findings in a readable form."""
    standards = result["standards"]
    if not standards:
        print("no standards or certificates found")
    else:
        print(f"found {len(standards)} standards and certificates:")
        for finding in standards:
            green = "  [green]" if finding["green"] else ""
            print(f"  [{finding['category']}] {finding['name']} "
                  f"(phase {finding['phase']}, seen {finding['count']}x){green}")
            print(f"      {finding['evidence']}")

    if result["green_signals"]:
        print("\ngreen-intent language:")
        for signal in result["green_signals"]:
            print(f"  {signal['keyword']} ({signal['count']}x)")


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

    result = find_standards(text, tables)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_report(result)


if __name__ == "__main__":
    main()