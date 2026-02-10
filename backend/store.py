"""
Vector store module.

Stores and retrieves embeddings using FAISS.
"""
import faiss

def create_index(embedding_dim):
    """Create a FAISS index for the given embedding dimension."""
    index = faiss.IndexFlatL2(embedding_dim)
    return index

def add_embeddings(index, embeddings):
    """Add embeddings to the FAISS index."""
    index.add(embeddings)

def save_index(index, path):
    """Save the FAISS index to disk."""
    faiss.write_index(index, path)

def load_index(path):
    """Load a FAISS index from disk."""
    index = faiss.read_index(path)
    return index

if __name__ == "__main__":
    import numpy as np

    dim = 384
    index = create_index(dim)

    vectors = np.random.rand(5, dim).astype("float32")
    add_embeddings(index, vectors)

    print("Vectors stored:", index.ntotal)