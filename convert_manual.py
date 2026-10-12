from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from tqdm import tqdm


# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PAGES_DIR,
    PROCESSED_DIR,
    JSONL_PATH,
    ERROR_LOG,
    MANIFEST_PATH,
    CHUNK_WORDS,
    OVERLAP_WORDS,
    MIN_TEXT_WORDS,
)


# ============================================================
# REGEX
# ============================================================

# Standard DTC:
# P1234
# B1234
# C1234
# U1234
#
# Ford-style suffix:
# C113A:11
# C113A:15

DTC_PATTERN = re.compile(
    r"\b([PBCU][0-9A-F]{4}(?::[0-9A-F]{2})?)\b",
    re.IGNORECASE,
)

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)

JUNK_PATTERN = re.compile(
    r"""
    (
        navigation|
        navbar|
        sidebar|
        breadcrumb|
        breadcrumbs|
        cookie|
        toolbar|
        print|
        share|
        search|
        searchbox|
        menu
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

CONTENT_PATTERN = re.compile(
    r"""
    (
        content|
        article|
        manual|
        document|
        page-content|
        main-content|
        repair|
        diagnosis|
        procedure
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+$",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# HTML ELEMENT IDENTITY
# ============================================================

def get_identity(tag: Tag) -> str:
    tag_id = tag.get("id", "")

    classes = tag.get("class", [])

    if not isinstance(classes, list):
        classes = [str(classes)]

    return f"{tag_id} {' '.join(str(x) for x in classes)}"


# ============================================================
# FIND MAIN CONTENT
# ============================================================

def find_main_content(soup: BeautifulSoup) -> Tag:

    # Remove obvious non-content tags
    for tag_name in [
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove semantic website UI
    for tag_name in [
        "nav",
        "header",
        "footer",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove likely navigation containers
    for tag in list(
        soup.find_all(
            ["div", "section", "aside"]
        )
    ):
        identity = get_identity(tag)

        if JUNK_PATTERN.search(identity):
            try:
                tag.decompose()
            except Exception:
                pass

    # Prefer <main> or <article>
    candidates = []

    candidates.extend(
        soup.find_all("main")
    )

    candidates.extend(
        soup.find_all("article")
    )

    # Then common containers
    candidates.extend(
        soup.find_all(
            ["div", "section"]
        )
    )

    if not candidates:
        return soup.body or soup

    best = None
    best_score = -1

    for tag in candidates:

        text = tag.get_text(
            " ",
            strip=True,
        )

        if len(text) < 100:
            continue

        identity = get_identity(tag)

        score = len(text)

        # Strong preference for actual content containers
        if CONTENT_PATTERN.search(identity):
            score += 100000

        # Penalize navigation-like elements
        if JUNK_PATTERN.search(identity):
            score -= 50000

        if score > best_score:
            best_score = score
            best = tag

    return best or soup.body or soup


# ============================================================
# TITLE
# ============================================================

def extract_title(
    soup: BeautifulSoup,
    main_content: Tag,
    html_path: Path,
) -> str:

    # H1
    h1 = main_content.find("h1")

    if h1:
        text = h1.get_text(
            " ",
            strip=True,
        )

        if text:
            return text

    # H2
    h2 = main_content.find("h2")

    if h2:
        text = h2.get_text(
            " ",
            strip=True,
        )

        if text:
            return text

    # HTML title
    if soup.title:

        text = soup.title.get_text(
            " ",
            strip=True,
        )

        if text:
            return text

    return html_path.stem


# ============================================================
# HEADINGS
# ============================================================

def extract_headings(
    main_content: Tag,
) -> list[str]:

    headings = []

    for tag in main_content.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]
    ):

        text = tag.get_text(
            " ",
            strip=True,
        )

        if text:
            headings.append(text)

    # Remove duplicate headings
    return list(
        dict.fromkeys(headings)
    )


# ============================================================
# DTC EXTRACTION
# ============================================================

def extract_dtcs(
    text: str,
) -> list[str]:

    codes = DTC_PATTERN.findall(text)

    return sorted(
        set(
            code.upper()
            for code in codes
        )
    )


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_images(
    main_content: Tag,
    html_path: Path,
) -> list[str]:

    images = []

    manual_root = PAGES_DIR.parent

    for img in main_content.find_all("img"):

        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or ""
        ).strip()

        if not src:
            continue

        # Ignore web/data images
        if src.startswith(
            (
                "http://",
                "https://",
                "data:",
            )
        ):
            continue

        try:

            resolved = (
                html_path.parent / src
            ).resolve()

            relative = resolved.relative_to(
                manual_root.resolve()
            )

            images.append(
                str(relative).replace(
                    "\\",
                    "/",
                )
            )

        except Exception:

            # Still preserve original path
            images.append(
                src.replace(
                    "\\",
                    "/",
                )
            )

    return list(
        dict.fromkeys(images)
    )


# ============================================================
# TABLE EXTRACTION
# ============================================================

def convert_tables(
    soup: BeautifulSoup,
) -> None:
    """
    Replace HTML tables with readable text.

    This is intentionally simple because automotive manuals
    frequently contain diagnostic tables where preserving the
    actual text matters more for retrieval than visual styling.
    """

    tables = soup.find_all("table")

    for table in tables:

        lines = []

        for row in table.find_all("tr"):

            cells = row.find_all(
                ["th", "td"]
            )

            values = []

            for cell in cells:

                value = cell.get_text(
                    " ",
                    strip=True,
                )

                value = re.sub(
                    r"\s+",
                    " ",
                    value,
                )

                if value:
                    values.append(value)

            if values:

                lines.append(
                    " | ".join(values)
                )

        if not lines:
            continue

        replacement = soup.new_tag(
            "div"
        )

        replacement.string = (
            "\nTABLE:\n"
            + "\n".join(lines)
            + "\nEND TABLE\n"
        )

        table.replace_with(
            replacement
        )


# ============================================================
# HTML TO MARKDOWN
# ============================================================

def html_to_markdown(
    html_path: Path,
):

    html = html_path.read_bytes()

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    main_content = find_main_content(
        soup
    )

    title = extract_title(
        soup,
        main_content,
        html_path,
    )

    headings = extract_headings(
        main_content
    )

    images = extract_images(
        main_content,
        html_path,
    )

    # Convert tables before Markdown conversion
    convert_tables(
        main_content
    )

    markdown = markdownify(
        str(main_content),
        heading_style="ATX",
        bullets="-",
    )

    markdown = normalize_text(
        markdown
    )

    metadata = {
        "title": title,
        "headings": headings,
        "images": images,
    }

    return markdown, metadata


# ============================================================
# SPLIT BY HEADINGS
# ============================================================

def split_by_headings(
    markdown: str,
) -> list[tuple[str, str]]:

    lines = markdown.splitlines()

    sections = []

    heading_stack = []

    current_lines = []

    def flush():

        text = "\n".join(
            current_lines
        ).strip()

        if text:

            heading_path = (
                " > ".join(
                    heading_stack
                )
            )

            sections.append(
                (
                    heading_path,
                    text,
                )
            )

        current_lines.clear()

    for line in lines:

        match = HEADING_PATTERN.match(
            line
        )

        if match:

            flush()

            level = len(
                match.group(1)
            )

            heading = match.group(
                2
            ).strip()

            heading_stack[:] = (
                heading_stack[
                    :level - 1
                ]
            )

            heading_stack.append(
                heading
            )

        else:

            current_lines.append(
                line
            )

    flush()

    if not sections and markdown.strip():

        sections.append(
            (
                "",
                markdown.strip(),
            )
        )

    return sections


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text: str,
    max_words: int,
    overlap_words: int,
) -> Iterable[str]:

    words = text.split()

    if len(words) <= max_words:
        yield text.strip()
        return

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text,
        )
        if p.strip()
    ]

    current = []
    current_count = 0

    for paragraph in paragraphs:

        paragraph_words = (
            paragraph.split()
        )

        # Huge individual paragraph
        if len(paragraph_words) > max_words:

            if current:

                chunk = "\n\n".join(
                    current
                ).strip()

                if chunk:
                    yield chunk

                current = []
                current_count = 0

            step = max(
                1,
                max_words - overlap_words,
            )

            for i in range(
                0,
                len(paragraph_words),
                step,
            ):

                piece = paragraph_words[
                    i:i + max_words
                ]

                if piece:
                    yield " ".join(
                        piece
                    ).strip()

            continue

        # Start a new chunk if needed
        if (
            current
            and current_count
            + len(paragraph_words)
            > max_words
        ):

            chunk = "\n\n".join(
                current
            ).strip()

            if chunk:
                yield chunk

            # Overlap
            if overlap_words > 0:

                previous_words = (
                    chunk.split()
                )

                overlap = previous_words[
                    -overlap_words:
                ]

                current = [
                    " ".join(overlap)
                ]

                current_count = len(
                    overlap
                )

            else:

                current = []
                current_count = 0

        current.append(
            paragraph
        )

        current_count += len(
            paragraph_words
        )

    if current:

        yield "\n\n".join(
            current
        ).strip()


# ============================================================
# PROCESS ONE HTML FILE
# ============================================================

def process_page(
    html_path: Path,
):

    markdown, metadata = (
        html_to_markdown(
            html_path
        )
    )

    if not markdown:
        return []

    if (
        len(markdown.split())
        < MIN_TEXT_WORDS
    ):
        return []

    source_file = str(
        html_path.relative_to(
            PAGES_DIR
        )
    ).replace(
        "\\",
        "/",
    )

    title = metadata[
        "title"
    ]

    headings = metadata[
        "headings"
    ]

    images = metadata[
        "images"
    ]

    page_hash = hashlib.sha1(
        source_file.encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    sections = split_by_headings(
        markdown
    )

    records = []

    for section_index, (
        heading_path,
        section_text,
    ) in enumerate(sections):

        chunks = chunk_text(
            section_text,
            CHUNK_WORDS,
            OVERLAP_WORDS,
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            if not chunk.strip():
                continue

            dtcs = extract_dtcs(
                chunk
            )

            header = (
                "Vehicle: Ford EcoSport\n"
                "Year: 2022\n"
                "Model: EcoSport Titanium\n"
                "Manual: Repair and Diagnosis\n"
                f"Title: {title}\n"
                f"Section: {heading_path}\n"
                f"Source page: {source_file}\n"
            )

            if images:

                image_info = (
                    "\nAssociated images:\n"
                    + "\n".join(images)
                    + "\n"
                )

            else:

                image_info = ""

            full_text = normalize_text(
                header
                + image_info
                + "\n"
                + chunk
            )

            chunk_id = (
                f"ecosport_"
                f"{page_hash}_"
                f"{section_index:04d}_"
                f"{chunk_index:04d}"
            )

            records.append(
                {
                    "id": chunk_id,
                    "vehicle": "Ford EcoSport",
                    "year": 2022,
                    "model": "EcoSport Titanium",
                    "manual": "Repair and Diagnosis",
                    "title": title,
                    "section": heading_path,
                    "source_file": source_file,
                    "dtc_codes": dtcs,
                    "images": images,
                    "headings": headings,
                    "text": full_text,
                }
            )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LEMON MANUAL CONVERTER")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Manual root:")
    print(PROJECT_ROOT / "2022 Ford EcoSport Titanium")

    print()
    print("HTML pages:")
    print(PAGES_DIR)

    # --------------------------------------------------------
    # Check pages folder
    # --------------------------------------------------------

    if not PAGES_DIR.exists():

        print()
        print("ERROR: pages folder not found.")
        print()
        print(
            "Expected:"
        )
        print(
            PAGES_DIR
        )

        return 1

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find HTML files
    # --------------------------------------------------------

    html_files = sorted(
        list(
            PAGES_DIR.rglob(
                "*.html"
            )
        )
        + list(
            PAGES_DIR.rglob(
                "*.htm"
            )
        )
    )

    print()
    print(
        f"Found {len(html_files):,} HTML files"
    )

    if not html_files:

        print(
            "No HTML files found."
        )

        return 1

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    processed_pages = 0
    total_chunks = 0
    errors = 0

    # --------------------------------------------------------
    # Deduplicate identical text
    # --------------------------------------------------------

    seen_hashes = set()

    # --------------------------------------------------------
    # Process sequentially
    # --------------------------------------------------------

    with (
        JSONL_PATH.open(
            "w",
            encoding="utf-8",
        ) as output,

        ERROR_LOG.open(
            "w",
            encoding="utf-8",
        ) as error_file,
    ):

        for html_path in tqdm(
            html_files,
            desc="Processing HTML",
            unit="page",
        ):

            try:

                records = process_page(
                    html_path
                )

                processed_pages += 1

                for record in records:

                    text_hash = (
                        hashlib.sha1(
                            record[
                                "text"
                            ].encode(
                                "utf-8",
                                errors="ignore",
                            )
                        ).hexdigest()
                    )

                    if (
                        text_hash
                        in seen_hashes
                    ):
                        continue

                    seen_hashes.add(
                        text_hash
                    )

                    output.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_chunks += 1

            except Exception as exc:

                errors += 1

                error_file.write(
                    f"{html_path}\t"
                    f"{type(exc).__name__}: "
                    f"{exc}\n"
                )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "vehicle": "Ford EcoSport",
        "year": 2022,
        "model": "EcoSport Titanium",
        "manual": "Repair and Diagnosis",
        "html_files_found": len(
            html_files
        ),
        "pages_processed": processed_pages,
        "chunks_created": total_chunks,
        "errors": errors,
        "jsonl_file": str(
            JSONL_PATH
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"HTML files found : "
        f"{len(html_files):,}"
    )

    print(
        f"Pages processed  : "
        f"{processed_pages:,}"
    )

    print(
        f"Chunks created   : "
        f"{total_chunks:,}"
    )

    print(
        f"Errors           : "
        f"{errors:,}"
    )

    print()
    print(
        f"JSONL:"
    )
    print(
        JSONL_PATH
    )

    print()
    print(
        f"Error log:"
    )
    print(
        ERROR_LOG
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )