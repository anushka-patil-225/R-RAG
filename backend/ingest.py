import os
from PyPDF2 import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "uploaded_docs")

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_documents(directory):
    """Load PDF and TXT documents from a directory."""
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            documents.append(text)

        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    return documents


def chunk_text(text, chunk_size=800, overlap=150):
    """
    Improved chunking that preserves paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    print("Loaded documents:", len(docs))
    if docs:
        chunks = chunk_text(docs[0])
        print("First document chunks:", len(chunks))