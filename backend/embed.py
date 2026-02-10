"""
Embedding module.

Uses sentence-transformers to convert text chunks into vectors.
"""

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_chunks):
    """Generate embeddings for a list of text chunks."""
    return model.encode(text_chunks)

if __name__ == "__main__":
    sample = ["This is a test sentence", "Another sentence"]
    embeddings = generate_embeddings(sample)
    print(embeddings.shape)
