"""
Vector store module.
Stores and retrieves embeddings using FAISS (Inner Product / cosine similarity).
"""

import faiss


def create_index(embedding_dim):
    index = faiss.IndexFlatIP(embedding_dim)
    return index


def add_embeddings(index, embeddings):
    index.add(embeddings)


def save_index(index, path):
    faiss.write_index(index, path)


def load_index(path):
    return faiss.read_index(path)
