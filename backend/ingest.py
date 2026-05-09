"""
Document ingestion module.
Handles PDF, DOCX, TXT extraction with structure-aware chunking.

Improvements over v1:
- Detects headings and section boundaries before chunking
- Keeps sections together — does NOT mix content across headings
- Preserves bullet-list blocks as atomic units
- Falls back to paragraph-overlap chunking within oversized sections
"""

import re


# ── Extractors ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_obj):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_obj)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def extract_text_from_docx(file_obj):
    from docx import Document
    doc = Document(file_obj)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(1, full_text)]


def extract_text_from_txt(file_obj):
    content = file_obj.read().decode("utf-8", errors="ignore")
    return [(1, content)]


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive blank lines
    text = re.sub(r"[^\x00-\x7F]+", " ", text)   # remove non-ASCII
    return text.strip()


# ── Structure Detection ────────────────────────────────────────────────────────

# Patterns that mark the start of a new semantic section
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+[\.\)]\s+[A-Z].{2,}"              # "1. Introduction" / "1) Title"
    r"|[A-Z][A-Z\s]{3,50}$"                # ALL-CAPS HEADING
    r"|(?:#{1,4})\s+.+"                    # Markdown ## Headings
    r"|[A-Z][^a-z\n]{0,60}:\s*$"           # "Section Title:"
    r")",
    re.MULTILINE
)


def _is_heading(line: str) -> bool:
    """Heuristically detect if a line is a section heading."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    # All-caps short line
    if line.isupper() and 3 <= len(line) <= 80:
        return True
    # Starts with a number and period/paren
    if re.match(r"^\d+[\.\)]\s+[A-Z]", line):
        return True
    # Markdown heading
    if re.match(r"^#{1,4}\s+", line):
        return True
    # Title-case short line ending with colon
    if line.endswith(":") and len(line) <= 60:
        return True
    return False


def split_into_sections(text: str) -> list:
    """
    Split text into [(heading_or_None, body_text)] sections.
    Sections are delimited by detected headings.
    """
    lines = text.split("\n")
    sections = []
    current_heading = None
    current_body = []

    for line in lines:
        if _is_heading(line):
            # Save previous section
            body = "\n".join(current_body).strip()
            if body or current_heading:
                sections.append((current_heading, body))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    # Flush last section
    body = "\n".join(current_body).strip()
    if body or current_heading:
        sections.append((current_heading, body))

    # If nothing was detected as a heading, return document as single section
    if not sections or (len(sections) == 1 and sections[0][0] is None):
        return [(None, text)]

    return sections


# ── Within-Section Chunking ────────────────────────────────────────────────────

def chunk_section(heading: str, body: str, chunk_size: int = 900, overlap: int = 150) -> list:
    """
    Chunk a single section's body text.
    - Bullet-list blocks are kept intact when possible.
    - Paragraph boundaries are preferred over mid-sentence splits.
    - heading is prepended to each chunk for context.
    """
    prefix = f"{heading}\n" if heading else ""
    full_text = prefix + body

    if len(full_text) <= chunk_size:
        return [full_text.strip()] if full_text.strip() else []

    # Split on double newlines (paragraph / list-block boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]
    chunks = []
    current_parts = []
    current_len = len(prefix)

    def flush(parts):
        text = (prefix + "\n\n".join(parts)).strip()
        if len(text) >= 40:
            chunks.append(text)

    def split_long_para(para):
        """Sentence-level fallback for paragraphs that exceed chunk_size."""
        sentences = re.split(r"(?<=[.!?])\s+", para)
        sub_parts, sub_len, sub_chunks = [], 0, []
        for sent in sentences:
            if sub_len + len(sent) > chunk_size and sub_parts:
                sub_chunks.append((prefix + " ".join(sub_parts)).strip())
                # overlap
                carry, ol = [], 0
                for s in reversed(sub_parts):
                    carry.insert(0, s)
                    ol += len(s)
                    if ol >= overlap:
                        break
                sub_parts = carry + [sent]
                sub_len = sum(len(s) for s in sub_parts)
            else:
                sub_parts.append(sent)
                sub_len += len(sent)
        if sub_parts:
            sub_chunks.append((prefix + " ".join(sub_parts)).strip())
        return sub_chunks

    for para in paragraphs:
        if len(prefix) + len(para) > chunk_size:
            if current_parts:
                flush(current_parts)
                current_parts, current_len = [], len(prefix)
            for sub in split_long_para(para):
                chunks.append(sub)
            continue

        if current_len + len(para) + 2 > chunk_size and current_parts:
            flush(current_parts)
            # Carry overlap
            carry, ol = [], 0
            for p in reversed(current_parts):
                carry.insert(0, p)
                ol += len(p)
                if ol >= overlap:
                    break
            current_parts = carry
            current_len = len(prefix) + sum(len(p) for p in current_parts)

        current_parts.append(para)
        current_len += len(para) + 2

    if current_parts:
        flush(current_parts)

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list:
    """
    Structure-aware chunking:
    1. Detect headings and split into sections.
    2. Chunk each section independently (never mixes sections).
    3. Return flat list of chunk strings.
    """
    sections = split_into_sections(text)
    all_chunks = []
    for heading, body in sections:
        if not body.strip():
            # Heading-only line — attach to next section's first chunk via prefix
            continue
        section_chunks = chunk_section(heading, body, chunk_size, overlap)
        all_chunks.extend(section_chunks)
    return all_chunks


def process_uploaded_file(file_obj, filename: str) -> list:
    """
    Process a single uploaded file and return list of chunk dicts.
    Each chunk: { text, doc_id, filename, page }
    """
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        pages = extract_text_from_pdf(file_obj)
    elif ext == "docx":
        pages = extract_text_from_docx(file_obj)
    elif ext == "txt":
        pages = extract_text_from_txt(file_obj)
    else:
        return []

    all_chunks = []
    doc_id = filename.replace(" ", "_").replace(".", "_")

    for page_num, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        for chunk in chunk_text(cleaned):
            all_chunks.append({
                "text":     chunk,
                "doc_id":   doc_id,
                "filename": filename,
                "page":     page_num,
            })

    return all_chunks
