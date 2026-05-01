"""
Document ingestion module.
Handles PDF, DOCX, TXT extraction and intelligent chunking with overlap.
Keeps structured records (e.g. employee/project blocks) intact when possible.
"""

import re


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


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)          # collapse horizontal whitespace only
    text = re.sub(r'\n{3,}', '\n\n', text)        # collapse 3+ blank lines to 2
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)    # remove non-ASCII
    return text.strip()


def chunk_text(text, chunk_size=900, overlap=150):
    """
    Character-budget chunking that respects paragraph/record boundaries.

    Strategy:
    1. Split on double-newlines (paragraph/record boundaries) first.
    2. Accumulate paragraphs into a chunk until the character budget is hit.
    3. When a chunk is emitted, carry the last `overlap` characters into the next chunk.
    4. If a single paragraph exceeds chunk_size, fall back to sentence-splitting within it.
    """
    # Split into paragraph blocks
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    chunks = []
    current_parts = []
    current_len = 0

    def flush(parts):
        chunk_str = "\n\n".join(parts).strip()
        if len(chunk_str) >= 40:
            chunks.append(chunk_str)

    def split_long_paragraph(para):
        """Sentence-split a paragraph that is itself too long."""
        sentences = re.split(r'(?<=[.!?])\s+', para)
        sub_parts = []
        sub_len = 0
        sub_chunks = []
        for sent in sentences:
            if sub_len + len(sent) > chunk_size and sub_parts:
                sub_chunks.append(" ".join(sub_parts))
                # overlap
                overlap_parts = []
                ol = 0
                for s in reversed(sub_parts):
                    overlap_parts.insert(0, s)
                    ol += len(s)
                    if ol >= overlap:
                        break
                sub_parts = overlap_parts + [sent]
                sub_len = sum(len(s) for s in sub_parts)
            else:
                sub_parts.append(sent)
                sub_len += len(sent)
        if sub_parts:
            sub_chunks.append(" ".join(sub_parts))
        return sub_chunks

    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush current buffer first
            if current_parts:
                flush(current_parts)
                current_parts = []
                current_len = 0
            # Then split the long paragraph itself
            for sub in split_long_paragraph(para):
                chunks.append(sub)
            continue

        if current_len + len(para) + 2 > chunk_size and current_parts:
            flush(current_parts)
            # Build overlap from tail of current_parts
            overlap_parts = []
            ol = 0
            for p in reversed(current_parts):
                overlap_parts.insert(0, p)
                ol += len(p)
                if ol >= overlap:
                    break
            current_parts = overlap_parts
            current_len = sum(len(p) for p in current_parts)

        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        flush(current_parts)

    return chunks


def process_uploaded_file(file_obj, filename):
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
                "text": chunk,
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num
            })

    return all_chunks
