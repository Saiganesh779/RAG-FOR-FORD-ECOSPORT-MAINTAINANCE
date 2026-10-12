import sys
import re
import requests
import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

sys.path.insert(0, ".")

from config import (
    DATABASE_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
)


# =========================================================
# SETTINGS
# =========================================================

TOP_K_EXACT = 8            # only used for DTCs we have no procedure map for
TOP_K_SEMANTIC = 2

MAX_CONTEXT_CHARS = 14000  # per LLM call (one pinpoint test per call)
NUM_CTX = 10240

INFO_CHUNK_CHARS = 3500    # cap for DTC chart / pinpoint info chunks
STEP_CHARS = 5000          # cap for one procedure step
FOLLOW_DEPTH = 10          # how many "GO to" hops to follow after the entry target
MAX_STEPS_PER_TEST = 30    # safety cap on steps expanded per pinpoint test
                           # override at run time: --depth=N  --max-steps=N

# The metadata key written by build_index.py is "source_file".
PAGE_KEYS = ("source_file", "source_page")


# =========================================================
# MANUAL MAP (verified from the manual)
# =========================================================

DTC_CHART_PAGE = "16051.html"

# Pinpoint test -> page describing the test and its DTC trigger conditions
INFO_PAGES = {
    "KF": "16053.html",
    "KN": "16054.html",
}

# Pinpoint test -> page holding the KF1/KF2... or KN1/KN2... procedure
PROCEDURE_PAGES = {
    "KF": "16184.html",
    "KN": "16193.html",
}

# (test, DTC base code) -> step reached from step 1 when that DTC is present
ENTRY_TARGETS = {
    ("KF", "P0480"): "KF3",
    ("KN", "P0480"): "KN4",
}

# Used only if the chart cannot be parsed
FALLBACK_ROUTES = {
    "P0480": ["KF", "KN"],
}

ANY_DTC_RE = re.compile(r"[PBCU][0-9A-F]{4}:[0-9A-F]{2}")


# =========================================================
# LOAD DATABASE
# =========================================================

print("Loading ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Loading embedding model...")

embed_model = SentenceTransformer(
    EMBED_MODEL
)

print("Ready.")


# =========================================================
# HELPERS
# =========================================================

def extract_dtcs(question):

    pattern = r"\b([PBCU][0-9A-F]{4}(?::[0-9A-F]{2})?)\b"

    matches = re.findall(
        pattern,
        question.upper()
    )

    return list(dict.fromkeys(matches))


def dtc_base(dtc):
    return dtc.split(":")[0].upper()


def get_source_page(metadata):

    if not metadata:
        return ""

    return (
        metadata.get("source_file")
        or metadata.get("source_page")
        or metadata.get("file")
        or ""
    )


def make_item(label, text, page, title, search_type, group="common"):

    return {
        "label": label,
        "document": text,
        "metadata": {
            "source_file": page,
            "title": title,
            "section": "",
        },
        "search_type": search_type,
        "group": group,
    }


def strip_header(document):
    """
    Every indexed chunk starts with a metadata header
    ('Vehicle: ... Source page: NNNN.html'). Remove it.
    """

    if not document.lstrip().startswith("Vehicle:"):
        return document

    return re.sub(
        r"\A\s*Vehicle:.*?Source page:[^\n]*\n+",
        "",
        document,
        count=1,
        flags=re.S,
    )


# =========================================================
# CHUNK ORDER
# =========================================================

def chunk_order(item):
    """
    Chunk ids written by convert_manual.py look like
        ecosport_<pagehash>_<section_index:04d>_<chunk_index:04d>
    chunk_index restarts at 0 inside every heading section, so the sort
    key MUST be (section_index, chunk_index). Sorting on chunk_index
    alone interleaves the sections of a page.
    """

    item_id = str(item.get("id", ""))
    metadata = item.get("metadata", {}) or {}

    m = re.search(r"_(\d+)_(\d+)$", item_id)

    if m:
        return (0, int(m.group(1)), int(m.group(2)), item_id)

    for key in ("chunk_index", "chunk_id", "position", "order", "index"):

        value = metadata.get(key)

        if value is not None:

            try:
                return (1, 0, int(value), item_id)
            except (ValueError, TypeError):
                pass

    return (2, 0, 0, item_id)


# =========================================================
# EXACT PAGE LOOKUP (metadata only, no $contains)
# =========================================================

_page_cache = {}


def get_page_chunks(page):
    """
    Return every chunk whose source_file metadata equals `page`,
    header-stripped and sorted into original order. Cached per run.
    """

    if page in _page_cache:
        return _page_cache[page]

    chunks = []

    for key in PAGE_KEYS:

        try:

            data = collection.get(
                where={key: page},
                include=["documents", "metadatas"],
            )

        except Exception as e:

            print(f"Could not load {page} using '{key}': {e}")
            continue

        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []

        for i, document in enumerate(documents):

            chunks.append({
                "id": ids[i] if i < len(ids) else f"{page}_{i}",
                "document": strip_header(document),
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas) and metadatas[i]
                    else {}
                ),
            })

        if chunks:
            break

    chunks.sort(key=chunk_order)

    if not chunks:
        print(
            f"WARNING: no chunks found for page {page} "
            f"(tried metadata keys {PAGE_KEYS})."
        )

    _page_cache[page] = chunks

    return chunks


def page_title(chunks):

    for chunk in chunks:

        title = chunk["metadata"].get("title")

        if title:
            return title

    return ""


# =========================================================
# STEP-AWARE EXTRACTION
# =========================================================

def heading_regex(prefix):
    """
    Matches a procedure step heading such as
        '3. KF3 CHECK THE FAN RELAY'
        'KN4 CHECK THE FCV CIRCUIT'
    but not a reference such as 'GO to KF3'.
    """

    return re.compile(
        rf"(?:\b\d+\.\s+)?(?<![Tt][Oo] )\b({prefix}\d+)\s+[A-Z]{{3,}}"
    )


def reflow_flat(text):
    """
    Long steps are split into sub-chunks whose text is whitespace-
    collapsed. Restore just enough line structure for the branch
    pruner: conditional bullets, questions and tables get their own lines.
    """

    if "\n" in text.strip():
        return text

    t = " " + text.strip()

    # '- For Ford GT, - FC1 Relay ...'  ->  own line for the condition
    t = re.sub(
        r"\s-\s(For\s+[^,\n]{1,80},)(?=\s-\s)",
        r"\n   - \1\n  ",
        t,
    )

    # 'Is the voltage greater than 10.5 V? TABLE:'  ->  own line
    t = re.sub(
        r"\s((?:Is|Are|Does|Do|Did|Was|Has|Have|Can)\b[^?\n]{3,200}\?)"
        r"\s+(?=TABLE:)",
        r"\n\n\1\n\n",
        t,
    )

    t = re.sub(r"\s*\bTABLE:\s*", "\nTABLE:\n     ", t)
    t = re.sub(r"\s*\bEND TABLE\s*", "\n     END TABLE\n", t)

    return t.strip("\n")


def merge_chunk_text(accumulated, following):
    """
    Append `following` (a continuation chunk) to `accumulated`.
    The continuation starts with a copy of the previous chunk's tail,
    often whitespace-collapsed, so the overlap is located with a
    whitespace-tolerant match and only the new text is appended.
    """

    acc_norm = re.sub(r"\s+", " ", accumulated).strip()
    window = following[:2500]

    for n in (160, 100, 60, 35):

        if len(acc_norm) < n:
            continue

        tail = acc_norm[-n:]

        if " " in tail:
            tail = tail.split(" ", 1)[1]   # drop a possibly cut first word

        words = tail.split()

        if len(words) < 3:
            continue

        pattern = r"\s+".join(re.escape(w) for w in words)

        m = re.search(pattern, window)

        if m:

            remainder = following[m.end():].strip()

            if not remainder:
                return accumulated

            return accumulated.rstrip() + "\n   " + reflow_flat(remainder)

    return accumulated.rstrip() + "\n" + reflow_flat(following.strip())


def cut_at_next_heading(text, own_id, pattern):
    """
    Return (text_up_to_next_other_step_heading, found_next_heading).
    """

    for m in pattern.finditer(text):

        if m.start() == 0:
            continue

        if m.group(1) != own_id:
            return text[:m.start()].rstrip(), True

    return text, False


def extract_step(chunks, step_id, prefix):
    """
    Locate the heading of `step_id` in the ordered chunks and return the
    text of that step: from its heading up to the next step heading,
    continuing across chunk boundaries. Returns None if not found.
    """

    pattern = heading_regex(prefix)

    start_index = None
    start_pos = 0

    for i, chunk in enumerate(chunks):

        for m in pattern.finditer(chunk["document"]):

            if m.group(1) == step_id:
                start_index = i
                start_pos = m.start()
                break

        if start_index is not None:
            break

    if start_index is None:
        return None

    text = chunks[start_index]["document"][start_pos:]

    text, done = cut_at_next_heading(text, step_id, pattern)

    j = start_index + 1

    while (
        not done
        and j < len(chunks)
        and len(text) < STEP_CHARS
    ):

        text = merge_chunk_text(text, chunks[j]["document"])

        text, done = cut_at_next_heading(text, step_id, pattern)

        j += 1

    return text[:STEP_CHARS]


# =========================================================
# DTC ROUTING FROM THE DTC CHART
# =========================================================

_discovery_cache = {}


def _scan(text, limit=600):
    """(page, title, document) for chunks whose text contains `text`."""

    try:

        data = collection.get(
            where_document={"$contains": text},
            limit=limit,
            include=["documents", "metadatas"],
        )

    except Exception as e:

        print(f"Discovery scan error for '{text}': {e}")
        return []

    out = []

    for document, metadata in zip(
        data.get("documents", []) or [],
        data.get("metadatas", []) or [],
    ):

        metadata = metadata or {}

        out.append((
            get_source_page(metadata),
            metadata.get("title", ""),
            strip_header(document),
        ))

    return out


def procedure_page_for(test):
    """
    Page holding '<test>1 ...' step headings. Uses the hand-written map
    first; otherwise discovers it from the index (cached).
    """

    if test in PROCEDURE_PAGES:
        return PROCEDURE_PAGES[test]

    key = ("proc", test)

    if key in _discovery_cache:
        return _discovery_cache[key]

    pattern = heading_regex(test)
    counts = {}

    for page, _title, document in _scan(f"{test}1 "):

        if page and any(
            m.group(1) == f"{test}1"
            for m in pattern.finditer(document)
        ):
            counts[page] = counts.get(page, 0) + 1

    page = max(counts, key=counts.get) if counts else None

    if page:
        print(f"Auto-discovered procedure page for test {test}: {page}")

    _discovery_cache[key] = page

    return page


def info_page_for(test):
    """Page with 'Possible Sources' / DTC trigger conditions for the test."""

    if test in INFO_PAGES:
        return INFO_PAGES[test]

    key = ("info", test)

    if key in _discovery_cache:
        return _discovery_cache[key]

    counts = {}

    for page, title, document in _scan(f"Pinpoint Test {test}:"):

        if not page or not title.startswith(f"Pinpoint Test {test}:"):
            continue

        if "Possible Sources" in document or "Trigger" in document:
            counts[page] = counts.get(page, 0) + 1

    page = max(counts, key=counts.get) if counts else None

    if page:
        print(f"Auto-discovered information page for test {test}: {page}")

    _discovery_cache[key] = page

    return page


def chart_pages_for_dtc(dtc):
    """The known PCM chart page, plus other DTC chart pages if needed."""

    base = dtc_base(dtc)

    if any(
        base in chunk["document"]
        for chunk in get_page_chunks(DTC_CHART_PAGE)
    ):
        return [DTC_CHART_PAGE]

    key = ("chart", base)

    if key in _discovery_cache:
        return _discovery_cache[key]

    pages = []

    for page, title, _doc in _scan(base, limit=1000):

        if (
            page
            and "DTC" in title
            and "Chart" in title
            and page not in pages
        ):
            pages.append(page)

    _discovery_cache[key] = pages

    return pages


def chart_chunks_for_dtc(dtc):

    base = dtc_base(dtc)

    return [
        chunk
        for page in chart_pages_for_dtc(dtc)
        for chunk in get_page_chunks(page)
        if base in chunk["document"]
    ]


def parse_routes(dtc, chart_chunks):
    """
    Read 'GO to Pinpoint Test XX' for this DTC from the chart rows.
    Each DTC occurrence is bounded by the next DTC code so rows that
    were flattened onto one line do not bleed into each other.
    """

    base = dtc_base(dtc)
    tests = []

    for chunk in chart_chunks:

        text = chunk["document"]

        matches = list(ANY_DTC_RE.finditer(text))

        for idx, m in enumerate(matches):

            if not m.group(0).startswith(base):
                continue

            end = (
                matches[idx + 1].start()
                if idx + 1 < len(matches)
                else len(text)
            )

            segment = text[m.start():end]

            for test in re.findall(
                r"Pinpoint\s+Test\s+([A-Z]{1,3})\b",
                segment,
            ):

                if test not in tests:
                    tests.append(test)

    if not tests:
        tests = list(FALLBACK_ROUTES.get(base, []))

    return tests


# =========================================================
# BRANCH APPLICABILITY
# =========================================================

# A bullet line such as "- For DTCs P028D or P028E," or "- For all others,"
COND_LINE_RE = re.compile(r"^\s*-\s*For\s+(.*?),\s*$", re.I)

ALL_OTHERS_LINE_RE = re.compile(
    r"^\s*-\s*For\s+all\s+others,\s*$", re.I | re.M
)

# Vehicles other than the EcoSport named in the manual's conditions
OTHER_VEHICLES_RE = re.compile(
    r"Ford GT|Bronco Sport|Expedition|Navigator|Transit Connect"
    r"|Nautilus|\bEdge\b|\bKA\b",
    re.I,
)

# Text that marks a "GO to" as NOT applicable to a present-DTC EcoSport
SKIP_MARKERS = re.compile(
    r"symptoms\s+without\s+DTCs?"
    r"|Ford GT|Bronco Sport|Expedition|Navigator|Transit Connect"
    r"|Nautilus|\bEdge\b|\bKA\b",
    re.I,
)

KEEP_MARKER = re.compile(r"all\s+others", re.I)


def prune_inapplicable_branches(text, dtc):
    """
    Inside a step, remove conditional blocks that cannot apply:
      - 'For DTCs X or Y,' blocks that do not list this DTC
      - the 'For all others,' block when a DTC-specific block matched
      - 'For <other vehicle>,' blocks when a 'For all others,' exists
    A block ends at the first non-indented text line (normally the
    question). A 'TABLE:' line sits at column 0 and never ends a block.
    """

    base = dtc_base(dtc)

    has_all_others = bool(ALL_OTHERS_LINE_RE.search(text))

    out = []
    skipping = False
    group_matched = False

    for line in text.split("\n"):

        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        m = COND_LINE_RE.match(line)

        if m:

            cond = m.group(1)
            codes = re.findall(r"[PBCU][0-9A-F]{4}", cond.upper())

            if codes:

                if base in codes:
                    skipping = False
                    group_matched = True
                    out.append(line)
                else:
                    skipping = True
                    out.append(
                        f"   [Branch for DTC(s) {', '.join(codes)} omitted: "
                        f"does not apply to {base}]"
                    )

                continue

            if KEEP_MARKER.match(cond):

                if group_matched:
                    skipping = True
                    out.append(
                        f"   [Branch 'For all others' omitted: a branch "
                        f"specific to {base} was already given]"
                    )
                else:
                    skipping = False
                    out.append(line)

                continue

            if has_all_others and OTHER_VEHICLES_RE.search(cond):

                skipping = True
                out.append(
                    f"   [Branch for {cond} omitted: other vehicle]"
                )
                continue

            skipping = False
            out.append(line)
            continue

        ends_block = (
            bool(stripped)
            and indent == 0
            and stripped != "TABLE:"
        )

        if skipping:

            if ends_block:
                skipping = False
            else:
                continue

        if ends_block:
            group_matched = False

        out.append(line)

    return "\n".join(out)


def next_step_refs(text, test):
    """
    'GO to <test><n>' references that apply to a present DTC on an
    EcoSport. A reference is skipped when the closest preceding marker is
    'symptoms without DTCs' or another vehicle; 'For all others' wins.
    """

    refs = []

    for m in re.finditer(rf"\bGO\s+to\s+({test}\d+)\b", text, re.I):

        window = text[max(0, m.start() - 120):m.start()]

        last_skip = max(
            (x.end() for x in SKIP_MARKERS.finditer(window)),
            default=-1,
        )

        last_keep = max(
            (x.end() for x in KEEP_MARKER.finditer(window)),
            default=-1,
        )

        if last_skip > last_keep:
            continue

        ref = m.group(1).upper()

        if ref not in refs:
            refs.append(ref)

    return refs


# =========================================================
# LOAD ONE PINPOINT TEST FOR A DTC
# =========================================================

def group_key(dtc, test):
    return f"{dtc_base(dtc)} / Pinpoint Test {test}"


def notes_item(group, notes):

    return make_item(
        "VERIFIED ROUTING FOR THIS TEST (built by the retrieval code)",
        "\n".join(notes),
        "",
        "",
        "routing",
        group=group,
    )


def step_number(step_id):
    digits = re.sub(r"\D", "", step_id)
    return int(digits) if digits else 0


def applicable_refs(text, dtc, test):
    """
    'GO to' targets that apply to this DTC: taken from the RESOLVED
    Yes/No outcomes when the step has a decision table, otherwise from
    next_step_refs on the raw text.
    """

    try:
        rendered = render_step(text, dtc, compact=True)
    except Exception:
        rendered = ""

    if re.match(rf"\s*{test}\d+:", rendered) and "Yes ->" in rendered:

        refs = []

        for m in re.finditer(rf"\bGO\s+to\s+({test}\d+)\b", rendered, re.I):

            ref = m.group(1).upper()

            if ref not in refs:
                refs.append(ref)

        return refs

    return next_step_refs(text, test)


def derive_entry_target(entry_text, dtc, test):
    """
    Read the entry step's Yes cell and resolve it for this DTC
    (DTC-specific clause, else 'For all others'). Returns e.g. 'KF3'.
    """

    if not entry_text:
        return None

    try:
        rendered = render_step(
            entry_text, dtc, only_yes=True, compact=True
        )
    except Exception:
        return None

    if not rendered.startswith(f"{test}1:"):
        return None

    m = re.search(rf"\bGO\s+to\s+({test}\d+)\b", rendered, re.I)

    return m.group(1).upper() if m else None


def load_test(test, dtc):

    base = dtc_base(dtc)
    group = group_key(dtc, test)

    items = []
    notes = []

    # ---- information page: trigger conditions / possible sources ----

    info_page = info_page_for(test)

    if info_page:

        chunks = get_page_chunks(info_page)

        selected = [
            c for c in chunks if base in c["document"]
        ][:2] or chunks[:1]

        for chunk in selected:

            items.append(make_item(
                f"Pinpoint Test {test}: DTC description, trigger "
                f"conditions, possible sources",
                chunk["document"][:INFO_CHUNK_CHARS],
                info_page,
                page_title(chunks),
                "exact_page",
                group=group,
            ))

    # ---- procedure page ----

    proc_page = procedure_page_for(test)

    if not proc_page:

        notes.append(
            f"Pinpoint Test {test}: procedure page is not mapped and "
            f"could not be discovered; procedure text unavailable."
        )

        return [notes_item(group, notes)] + items

    chunks = get_page_chunks(proc_page)

    if not chunks:

        notes.append(
            f"Pinpoint Test {test}: procedure page {proc_page} "
            f"could not be loaded from the index."
        )

        return [notes_item(group, notes)] + items

    title = page_title(chunks)

    entry_id = f"{test}1"

    entry_text = extract_step(chunks, entry_id, test)

    items.append(make_item(
        f"{entry_id} (entry step, check for DTCs)",
        entry_text
        if entry_text is not None
        else (
            f"STEP {entry_id} WAS NOT FOUND in the indexed chunks of "
            f"{proc_page}. Do not infer it."
        ),
        proc_page,
        title,
        "procedure_step",
        group=group,
    ))

    # ---- which step does the entry step send this DTC to? ----
    # Priority: hand-verified map, then derived from the entry step's
    # Yes cell, then every applicable 'GO to' in the entry step.

    map_target = ENTRY_TARGETS.get((test, base))
    derived = derive_entry_target(entry_text, dtc, test)
    target = map_target or derived

    if target:

        confirmed = bool(
            entry_text
            and re.search(rf"GO\s+to\s+{target}\b", entry_text, re.I)
        )

        source = "manual notes" if map_target else "parsed from the step text"

        notes.append(
            f"{base} -> Pinpoint Test {test} -> {entry_id}: {base} present "
            f"-> GO to {target} ("
            + (
                f"{entry_id} text confirms this"
                if confirmed
                else f"NOT confirmed in retrieved {entry_id} text; "
                     f"target from {source}"
            )
            + ")."
        )

        if map_target and derived and derived != map_target:

            notes.append(
                f"WARNING: parsing {entry_id} gives {derived}, but the "
                f"manual notes say {map_target}. Check the manual page."
            )

        frontier = [(target, entry_id)]

    else:

        frontier = [
            (ref, entry_id)
            for ref in (
                next_step_refs(entry_text, test) if entry_text else []
            )
        ]

        notes.append(
            f"{base} -> Pinpoint Test {test} -> {entry_id}: no entry "
            f"target could be resolved; applicable GO to references from "
            f"{entry_id} are included."
        )

    # ---- walk the tree: breadth-first discovery, capped ----

    seen = {entry_id}
    found = {}          # step_id -> (text, parent, depth)
    capped = []
    level = frontier
    depth = 0

    while level and depth <= FOLLOW_DEPTH:

        next_level = []

        for step_id, parent in level:

            if step_id in seen:
                continue

            if len(found) >= MAX_STEPS_PER_TEST:
                capped.append(step_id)
                continue

            seen.add(step_id)

            text = extract_step(chunks, step_id, test)

            if text:
                text = prune_inapplicable_branches(text, dtc)

            found[step_id] = (text, parent, depth)

            if text:
                for ref in applicable_refs(text, dtc, test):
                    next_level.append((ref, step_id))

        level = next_level
        depth += 1

    unexpanded = list(dict.fromkeys(
        capped + [s for s, _ in level if s not in seen]
    ))

    if unexpanded:

        notes.append(
            f"NOT EXPANDED (limit reached): {', '.join(unexpanded)}. "
            f"Re-run with --depth=N / --max-steps=N to include them."
        )

    # ---- emit in manual order (by step number), not discovery order ----

    for step_id in sorted(found, key=step_number):

        text, parent, step_depth = found[step_id]

        if step_depth == 0:
            label = f"{step_id} (step reached from {entry_id} for {base})"
        else:
            label = f"{step_id} (referenced by GO to in {parent})"

        items.append(make_item(
            label,
            text
            if text is not None
            else (
                f"STEP {step_id} WAS NOT FOUND in the indexed chunks of "
                f"{proc_page}. Do not infer it."
            ),
            proc_page,
            title,
            "procedure_step",
            group=group,
        ))

    return [notes_item(group, notes)] + items


# =========================================================
# FALLBACK EXACT SEARCH (DTCs without a procedure map)
# =========================================================

def exact_search(dtc):

    results = []

    try:

        data = collection.get(
            where_document={"$contains": dtc_base(dtc)},
            limit=TOP_K_EXACT,
            include=["documents", "metadatas"],
        )

        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []

        for i, document in enumerate(documents):

            results.append({
                "label": "DTC text match",
                "id": ids[i] if i < len(ids) else f"exact_{dtc}_{i}",
                "document": strip_header(document),
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas) and metadatas[i]
                    else {}
                ),
                "search_type": "exact",
                "group": "common",
            })

    except Exception as e:

        print(f"Exact search error for {dtc}: {e}")

    return results


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(question):

    results = []

    try:

        embedding = embed_model.encode(question).tolist()

        data = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K_SEMANTIC,
            include=["documents", "metadatas", "distances"],
        )

        documents = data.get("documents", [[]])[0]
        metadatas = data.get("metadatas", [[]])[0]
        distances = data.get("distances", [[]])[0]
        ids = data.get("ids", [[]])[0]

        for i, document in enumerate(documents):

            results.append({
                "label": "Semantic match",
                "id": ids[i] if i < len(ids) else f"semantic_{i}",
                "document": strip_header(document),
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas) and metadatas[i]
                    else {}
                ),
                "distance": (
                    distances[i] if i < len(distances) else None
                ),
                "search_type": "semantic",
                "group": "common",
            })

    except Exception as e:

        print(f"Semantic search error: {e}")

    return results


# =========================================================
# RETRIEVE
# =========================================================

def retrieve(question):

    dtcs = extract_dtcs(question)

    if not dtcs:
        return semantic_search(question)

    print(f"Detected DTC(s): {', '.join(dtcs)}")

    common = []
    grouped = []
    fallback = []
    route_notes = []

    for dtc in dtcs:

        # 1. DTC chart rows for this DTC (exact page, filtered by code)

        chart = chart_chunks_for_dtc(dtc)

        for chunk in chart[:2]:

            chart_page = get_source_page(chunk["metadata"]) or DTC_CHART_PAGE

            common.append(make_item(
                f"DTC chart rows for {dtc_base(dtc)}",
                chunk["document"][:INFO_CHUNK_CHARS],
                chart_page,
                page_title(get_page_chunks(chart_page)),
                "exact_page",
            ))

        # 2. Which pinpoint tests does the chart route this DTC to?

        tests = parse_routes(dtc, chart)

        known = [t for t in tests if procedure_page_for(t)]

        if not known:

            fallback.extend(exact_search(dtc))
            continue

        route_notes.append(
            f"{dtc_base(dtc)} routes to Pinpoint Test(s): "
            f"{', '.join(tests)}. These are all applicable; they are "
            f"not Yes/No alternatives."
        )

        # 3. Load each test's info + step-aware procedure

        for test in known:

            print(f"Loading {dtc_base(dtc)} {test} procedure...")

            for loaded in load_test(test, dtc):
                loaded["dtc"] = dtc
                grouped.append(loaded)

    if not common and not grouped:
        return fallback or semantic_search(question)

    results = []

    if route_notes:

        results.append(make_item(
            "VERIFIED ROUTING (built by the retrieval code)",
            "\n".join(route_notes),
            "",
            "",
            "routing",
        ))

    results.extend(common)
    results.extend(grouped)
    results.extend(fallback)

    return results


# =========================================================
# DETERMINISTIC PATH RENDERING
# =========================================================

STEP_TITLE_RE = re.compile(r"\s*(?:\d+\.\s+)?([A-Z]{2,3}\d+)\s+(.*\S)\s*$")
YESNO_RE = re.compile(r"\s*Yes\s*\|\s*No\b(.*)$")


def tidy(text):
    text = " ".join(text.split())
    return re.sub(r"\s+\.", ".", text)


# A clause is "For <condition>, <action>". The action starts at the first
# comma followed by an action word, so lists such as
# "For DTCs P028D, P028E, or P0480, GO to KF36" keep their whole condition.
ACTION_START_RE = re.compile(
    r",\s+(?=(?:GO|REPAIR|INSTALL|RETURN|REPLACE|CARRY|REFER|CLEAR|The|If)\b)"
)


def resolve_cell(cell, dtc):
    """
    Resolve a Yes/No table cell for this DTC on an EcoSport.
    Applicable clauses: those naming this DTC and those naming EcoSport.
    If several apply they are ALL returned (the manual does not give a
    priority), otherwise 'For all others'. Other vehicles and
    'symptoms without DTCs' clauses are dropped.
    """

    base = dtc_base(dtc)

    text = " ".join(cell.split())

    clauses = re.split(r"(?<=\.)\s+(?=For\b)", text)

    if len(clauses) == 1 and not text.startswith("For "):
        return tidy(text)

    applicable = []
    others = None

    for clause in clauses:

        parts = ACTION_START_RE.split(clause, maxsplit=1)

        m = re.match(r"For\s+(.*)$", parts[0])

        if len(parts) < 2 or not m:
            continue

        cond = m.group(1).strip()
        action = parts[1]

        codes = re.findall(r"[PBCU][0-9A-F]{4}", cond.upper())

        if codes:

            if base in codes:
                applicable.append((cond, action))

        elif re.fullmatch(r"all\s+others", cond, re.I):
            others = action

        elif (
            re.search(r"\bEcoSport\b", cond, re.I)
            and not re.search(r"symptoms\s+without", cond, re.I)
        ):
            applicable.append((cond, action))

    if len(applicable) == 1:
        return tidy(applicable[0][1])

    if len(applicable) > 1:

        return (
            tidy(" ".join(f"For {c}, {a}" for c, a in applicable))
            + " [more than one branch matches; the manual text does "
              "not say which takes priority]"
        )

    if others:
        return tidy(others)

    return tidy(text)


def clean_action_lines(lines):

    out = []

    for line in lines:

        s = line.strip()

        if not s or s in ("TABLE:", "END TABLE"):
            continue

        if s.startswith("[Branch "):
            continue

        if re.fullmatch(r"-\s*For\s+all\s+others,", s, re.I):
            continue

        if len(s) > 120:
            parts = re.split(r"(?<=[.:])\s-\s(?=[A-Z])", s)
        else:
            parts = [s]

        for i, part in enumerate(parts):

            part = part.strip()

            out.append(part if i == 0 else "- " + part)

    return out


def render_step(text, dtc, only_yes=False, compact=False):
    """
    Turn one extracted step into a clean block: title, actions, the
    question, and the Yes/No outcomes resolved for this DTC.
    """

    lines = text.split("\n")

    m = STEP_TITLE_RE.match(lines[0])

    if not m:
        return text

    step_id, title = m.group(1), m.group(2)

    head = f"STEP {step_id}: {title}"

    idx = next(
        (i for i, l in enumerate(lines) if YESNO_RE.match(l)),
        None,
    )

    if idx is None:

        if compact:
            return f"{step_id}: {title}."

        body = clean_action_lines(lines[1:])

        return "\n".join(
            [head]
            + ["    " + l for l in body]
            + ["    (no Yes/No decision table found in this step)"]
        )

    t_start = (
        idx - 1
        if idx > 0 and lines[idx - 1].strip() == "TABLE:"
        else idx
    )

    body = lines[1:t_start]

    question = None

    for k in range(len(body) - 1, -1, -1):

        if body[k].strip():

            if body[k].strip().endswith("?"):
                question = body[k].strip()
                body = body[:k]

            break

    rows = []

    rest = YESNO_RE.match(lines[idx]).group(1).strip()

    if rest:
        rows.append(rest)

    for l in lines[idx + 1:]:

        if l.strip() == "END TABLE":
            break

        if l.strip():
            rows.append(l.strip())

    yes_parts = []
    no_parts = []

    for row in rows:

        parts = row.split(" | ")

        if len(parts) >= 2:
            yes_parts.append(parts[0])
            no_parts.append(" | ".join(parts[1:]))
        else:
            yes_parts.append(row)

    yes_text = resolve_cell(" ".join(yes_parts), dtc)
    no_text = resolve_cell(" ".join(no_parts), dtc)

    if compact:

        if only_yes:
            return (
                f"{step_id}: {title}. "
                f"{dtc_base(dtc)} present -> {yes_text}"
            )

        return (
            f"{step_id}: {title}. Yes -> {yes_text} No -> {no_text}"
        )

    out = [head]

    out += ["    " + l for l in clean_action_lines(body)]

    if question:
        out.append(f"    Question: {question}")

    if only_yes:
        out.append(f"    {dtc_base(dtc)} present -> {yes_text}")
    else:
        out.append(f"    Yes -> {yes_text}")
        out.append(f"    No  -> {no_text}")

    return "\n".join(out)


_full_text_cache = {}


def get_full_page_text(page):
    """All chunks of a page merged into one text (overlap removed)."""

    if page in _full_text_cache:
        return _full_text_cache[page]

    text = ""

    for chunk in get_page_chunks(page):

        text = (
            chunk["document"]
            if not text
            else merge_chunk_text(text, chunk["document"])
        )

    _full_text_cache[page] = text

    return text


def info_summary(test, dtc):
    """DTC description, trigger condition and possible sources, verbatim."""

    base = dtc_base(dtc)
    page = info_page_for(test)

    if not page:
        return []

    text = get_full_page_text(page)

    lines = [f"Information page: {page}"]

    m = re.search(
        rf"^.*?{base}:\d\d\s*\|\s*([^|\n]+?)\s*\|\s*(.+)$",
        text,
        re.M,
    )

    if m:
        lines.append(f"DTC: {base} - {m.group(1).strip()}")
        lines.append(f"Trigger condition: {tidy(m.group(2))}")
    else:
        lines.append(f"DTC {base}: trigger condition not found on page.")

    ps = re.search(
        r"Possible Sources\s*\n((?:[ \t]*-[ \t]*.+\n?)+)",
        text,
    )

    if ps:

        sources = [
            s.strip().lstrip("-").strip()
            for s in ps.group(1).strip().split("\n")
            if s.strip()
        ]

        lines.append("Possible sources: " + "; ".join(sources))

    else:
        lines.append("Possible sources: not found on page.")

    return lines


def render_group(group, items, full=True):

    test = group.rsplit(" ", 1)[-1]

    dtc = next((i.get("dtc") for i in items if i.get("dtc")), "")

    out = [f"Pinpoint Test {test} (procedure page {procedure_page_for(test) or '?'})"]

    for item in items:

        if item.get("search_type") == "routing":
            out.append("Routing: " + item["document"].replace("\n", " "))

    out += info_summary(test, dtc)

    out.append("")
    out.append("Decision flow (built by code from the manual text):")

    for item in items:

        if item.get("search_type") != "procedure_step":
            continue

        document = item["document"]

        if document.startswith("STEP "):
            out.append("  " + document)
            continue

        out.append(
            "  "
            + render_step(
                document,
                dtc,
                only_yes="entry step" in item.get("label", ""),
                compact=True,
            )
        )

    if full:

        out.append("")
        out.append(
            "Diagnostic path (from the manual; only the branches that apply "
            "to this DTC on an EcoSport):"
        )
        out.append("")

        for item in items:

            if item.get("search_type") != "procedure_step":
                continue

            document = item["document"]

            if document.startswith("STEP "):
                out.append(document)
                out.append("")
                continue

            out.append(
                render_step(
                    document,
                    dtc,
                    only_yes="entry step" in item.get("label", ""),
                )
            )

            out.append("")

    return "\n".join(out).rstrip()


# =========================================================
# DISPLAY SOURCES
# =========================================================

def show_sources(results):

    print()
    print("=" * 70)
    print("RETRIEVED MANUAL SOURCES")
    print("=" * 70)

    for index, item in enumerate(results, start=1):

        metadata = item.get("metadata", {})

        print()
        print(f"[SOURCE {index}]")
        print(f"Label  : {item.get('label', '')}")

        group = item.get("group", "common")

        if group != "common":
            print(f"Group  : {group}")

        page = get_source_page(metadata)

        if page:
            print(f"File   : {page}")
            print(f"Title  : {metadata.get('title', '')}")

        print(f"Search : {item.get('search_type', '')}")

        document = item.get("document", "")

        if document:

            print()
            print("CONTENT PREVIEW:")
            print("-" * 70)
            print(document[:1200])

            if len(document) > 1200:
                print("...")

            print("-" * 70)


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(question, results):

    parts = []
    total = 0

    for index, item in enumerate(results, start=1):

        metadata = item.get("metadata", {})
        page = get_source_page(metadata)

        header = f"[SOURCE {index}] {item.get('label', '')}\n"

        if page:
            header += (
                f"File: {page}\n"
                f"Title: {metadata.get('title', '')}\n"
            )

        block = header + "MANUAL CONTENT:\n" + item.get("document", "")

        remaining = MAX_CONTEXT_CHARS - total

        if remaining <= 0:
            break

        if len(block) > remaining:
            parts.append(block[:remaining])
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


# =========================================================
# OLLAMA
# =========================================================

SYSTEM_PROMPT_SUMMARY = (
    "You are a Ford EcoSport 2022 Repair and Diagnosis service manual "
    "assistant.\n\n"

    "You are given a diagnostic path that was already extracted from "
    "the manual and resolved for this DTC. Explain it to a technician "
    "in plain English.\n\n"

    "RULES:\n"
    "1. Use ONLY the supplied text.\n"
    "2. Do not add, remove, reorder or rename steps. Mention only the "
    "step ids that appear (for example KF3).\n"
    "3. Keep every Yes and No outcome and every GO to target exactly "
    "as given.\n"
    "4. Do not give outside automotive advice.\n"
    "5. Summarize measurements in one short phrase; do not list every "
    "connector.\n\n"

    "FORMAT:\n"
    "One sentence overview.\n"
    "Then one line per step: <id>: <what is checked>. Yes -> <next>. "
    "No -> <next>.\n"
)

SYSTEM_PROMPT_GENERAL = (
    "You are a Ford EcoSport 2022 Repair and Diagnosis service manual "
    "assistant.\n\n"
    "Use ONLY the supplied manual content. Do not use outside "
    "automotive knowledge. Do not guess. Do not invent steps.\n"
    "Follow explicit GO to instructions and keep Yes/No branches "
    "separate. Quote Ford terminology, part numbers and values exactly.\n"
    "Name the source file for each fact. If the answer is not in the "
    "supplied content, say so.\n"
)


def ask_ollama(question, context, focus=None):

    if focus:

        system_prompt = SYSTEM_PROMPT_SUMMARY

        user_prompt = (
            "QUESTION:\n"
            + question
            + "\n\n"
            + f"Summarize ONLY: {focus}.\n\n"
            + "EXTRACTED DIAGNOSTIC PATH:\n"
            + context
            + "\n\n"
            + "Summarize this path in the required format."
        )

    else:

        system_prompt = SYSTEM_PROMPT_GENERAL

        user_prompt = (
            "QUESTION:\n"
            + question
            + "\n\n"
            + "RETRIEVED MANUAL CONTENT:\n"
            + context
            + "\n\n"
            + "Answer ONLY from the retrieved manual content."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": NUM_CTX,
        },
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600,
        )

        response.raise_for_status()

        answer = response.json().get("response", "")

        if not answer:
            return "Ollama returned an empty response."

        return answer.strip()

    except requests.exceptions.RequestException as e:

        return "Could not connect to Ollama:\n" + str(e)

    except Exception as e:

        return "Ollama error:\n" + str(e)


# =========================================================
# DEBUG: verify chunk order and step headings
# =========================================================

def debug_pages():

    for test, page in PROCEDURE_PAGES.items():

        chunks = get_page_chunks(page)

        print()
        print("=" * 70)
        print(f"{page} (Pinpoint Test {test}): {len(chunks)} chunks")
        print("=" * 70)

        if not chunks:
            continue

        print("Metadata keys:", sorted(chunks[0]["metadata"].keys()))

        pattern = heading_regex(test)

        for i, chunk in enumerate(chunks):

            headings = [
                m.group(1) for m in pattern.finditer(chunk["document"])
            ]

            print(
                f"[{i}] id={chunk['id']} len={len(chunk['document'])} "
                f"headings={headings}"
            )

        steps = [f"{test}1"] + [
            t for (tt, _), t in ENTRY_TARGETS.items() if tt == test
        ]

        steps += ["KF6", "KN6", "KN9"] if test == "KN" else []

        for step in steps:

            if not step.startswith(test):
                continue

            text = extract_step(chunks, step, test)

            print()
            print(f"--- extract_step({step}) ---")
            print(text if text else "NOT FOUND")


# =========================================================
# MAIN
# =========================================================

def main():

    args = sys.argv[1:]

    global FOLLOW_DEPTH, MAX_STEPS_PER_TEST

    flags = {a for a in args if a.startswith("--")}
    words = [a for a in args if not a.startswith("--")]

    for flag in flags:

        if flag.startswith("--depth="):
            FOLLOW_DEPTH = int(flag.split("=", 1)[1])

        elif flag.startswith("--max-steps="):
            MAX_STEPS_PER_TEST = int(flag.split("=", 1)[1])

    if "--debug" in flags:
        debug_pages()
        return

    if words:
        question = " ".join(words)
    else:
        question = input("Question: ").strip()

    if not question:
        print("No question provided.")
        return

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    results = retrieve(question)

    if not results:
        print()
        print("No relevant manual content was retrieved.")
        return

    if "--no-sources" not in flags:
        show_sources(results)

    groups = []

    for item in results:

        group = item.get("group", "common")

        if group != "common" and group not in groups:
            groups.append(group)

    # ---------------- non-DTC question: normal RAG ----------------

    if not groups:

        context = build_context(question, results)

        print()
        print("=" * 70)
        print("OLLAMA ANSWER")
        print("=" * 70)
        print(f"(context size: {len(context):,} characters)")
        print()
        print(ask_ollama(question, context))

        return

    # ---------------- DTC question: deterministic path ----------------

    for group in groups:

        items = [
            item for item in results
            if item.get("group") == group
        ]

        rendered = render_group(group, items)

        print()
        print("#" * 70)
        print(f"# {group}")
        print("#" * 70)
        print()
        print(rendered)

        if "--explain" not in flags:
            continue

        print()
        print("-" * 70)
        print("AI explanation (--explain): written by a small local model")
        print("and can contain mistakes. Rely on the decision flow and the")
        print("path above, not on this text.")
        print("-" * 70)
        print()
        print(ask_ollama(
            question,
            render_group(group, items, full=False),
            focus=group,
        ))

def answer_question(question, explain=True):
    results = retrieve(question)

    if not results:
        return "No relevant manual content was retrieved."

    groups = []

    for item in results:
        group = item.get("group", "common")

        if group != "common" and group not in groups:
            groups.append(group)

    # Normal non-DTC question
    if not groups:
        context = build_context(question, results)
        return ask_ollama(question, context)

    # DTC question
    answers = []

    for group in groups:
        items = [
            item for item in results
            if item.get("group") == group
        ]

        rendered = render_group(group, items)

        if explain:
            explanation = ask_ollama(
                question,
                render_group(group, items, full=False),
                focus=group
            )

            answers.append(
                rendered
                + "\n\nAI explanation:\n"
                + explanation
            )
        else:
            answers.append(rendered)

    return "\n\n".join(answers)
if __name__ == "__main__":
    main()