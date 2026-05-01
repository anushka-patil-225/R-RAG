"""
Embedding module.
Uses sentence-transformers to convert text chunks into vectors.
Model is loaded once and cached.
"""

import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def generate_embeddings(text_chunks):
    model = get_model()
    embeddings = model.encode(
        text_chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    )
    return np.array(embeddings).astype("float32")
