"""
Retrieval module.
Performs semantic search + embedding reranking for high-quality chunk selection.
"""

import numpy as np
from backend.embed import generate_embeddings


def retrieve_top_k(query, index, documents, k=5):
    """
    Retrieve top-k most relevant chunks using FAISS + reranking.
    Returns list of chunk dicts sorted by relevance.
    """
    if not documents or index is None:
        return []

    query_embedding = generate_embeddings([query])[0]

    # Over-fetch then rerank (3x ratio)
    fetch_k = min(k * 3, len(documents))
    distances, indices = index.search(
        np.array([query_embedding]), fetch_k
    )

    candidates = []
    seen_keys = set()

    for i in indices[0]:
        if i < 0 or i >= len(documents):
            continue
        chunk = documents[i]
        text = chunk["text"]
        key = text[:200]
        if key in seen_keys or len(text) < 40:
            continue
        seen_keys.add(key)
        candidates.append(chunk)

    if not candidates:
        return []

    # Rerank using cosine similarity
    candidate_texts = [c["text"] for c in candidates]
    candidate_embeddings = generate_embeddings(candidate_texts)

    scores = [
        (float(np.dot(query_embedding, emb)), i)
        for i, emb in enumerate(candidate_embeddings)
    ]
    scores.sort(reverse=True, key=lambda x: x[0])

    # Apply score threshold: always return at least top 2
    MIN_SCORE = 0.25
    top_scores = scores[:k]
    above_threshold = [(score, idx) for score, idx in top_scores if score >= MIN_SCORE]
    if not above_threshold:
        above_threshold = top_scores[:2]

    results = []
    for score, idx in above_threshold:
        c = candidates[idx]
        results.append({
            "text": c["text"],
            "doc_id": c["doc_id"],
            "filename": c.get("filename", c["doc_id"]),
            "page": c.get("page", 1),
            "score": round(score, 4)
        })

    return results
