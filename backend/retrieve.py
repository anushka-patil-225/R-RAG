"""
Retrieval module.
Semantic search + cosine reranking for high-quality chunk selection.

Improvements over v1:
- Optional LLM-based query expansion for richer recall
- Keyword boost: chunks containing query terms are scored slightly higher
- Cleaner deduplication and score thresholding
"""

import re
import numpy as np
from backend.embed import generate_embeddings

# Toggle query expansion (adds one LLM call per top-level query).
# Set to False to save tokens when using small/offline models.
ENABLE_QUERY_EXPANSION = False


# ── Query Expansion ────────────────────────────────────────────────────────────

def expand_query(query: str) -> list:
    """
    Use the LLM to generate 1-2 alternative phrasings of the query.
    Returns a list of queries (original + expansions), deduplicated.
    """
    # Import here to avoid circular imports at module load
    from backend.llm import call_llm

    prompt = f"""Rephrase this question in 1-2 alternative ways that preserve its exact meaning.
The rephrases should use different vocabulary but ask for the same information.

Question: {query}

Return ONLY a numbered list of rephrased questions. No explanation."""

    result = call_llm(prompt, max_tokens=150)
    if not result or "❌" in result:
        return [query]

    expansions = [query]
    for line in result.strip().split("\n"):
        cleaned = re.sub(r"^[\d\.\)\-\•\*]+\s*", "", line.strip()).strip()
        if cleaned and len(cleaned) > 8 and cleaned.lower() != query.lower():
            expansions.append(cleaned)
            if len(expansions) >= 3:   # original + 2 expansions max
                break

    return expansions


# ── Keyword Boost ──────────────────────────────────────────────────────────────

def _keyword_boost(query: str, text: str, base_score: float, boost: float = 0.03) -> float:
    """Add a small boost for chunks that contain key query terms."""
    query_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
    text_lower = text.lower()
    matches = sum(1 for term in query_terms if term in text_lower)
    if not query_terms:
        return base_score
    hit_ratio = matches / len(query_terms)
    return base_score + boost * hit_ratio


# ── Main Retrieval Function ────────────────────────────────────────────────────

def retrieve_top_k(query: str, index, documents: list, k: int = 5) -> list:
    """
    Retrieve top-k most relevant chunks using FAISS + cosine reranking.

    Pipeline:
    1. (Optional) expand query into multiple phrasings
    2. Embed all query variants
    3. Over-fetch candidates from FAISS
    4. Rerank using cosine similarity + keyword boost
    5. Apply score threshold; always return at least 2 results if available

    Returns list of chunk dicts sorted by relevance score (desc).
    """
    if not documents or index is None:
        return []

    # ── Step 1: Query variants ────────────────────────────────────────────────
    if ENABLE_QUERY_EXPANSION:
        queries = expand_query(query)
    else:
        queries = [query]

    # ── Step 2: Embed all queries ─────────────────────────────────────────────
    query_embeddings = generate_embeddings(queries)  # shape: (n_queries, dim)

    # Average embeddings across variants for a combined query vector
    combined_query = np.mean(query_embeddings, axis=0).astype("float32")
    # Re-normalise after averaging (embeddings were normalised per-vector)
    norm = np.linalg.norm(combined_query)
    if norm > 0:
        combined_query = combined_query / norm

    # ── Step 3: FAISS over-fetch ──────────────────────────────────────────────
    fetch_k = min(k * 3, len(documents))
    distances, indices = index.search(np.array([combined_query]), fetch_k)

    candidates = []
    seen_keys = set()

    for i in indices[0]:
        if i < 0 or i >= len(documents):
            continue
        chunk = documents[i]
        key = chunk["text"][:200]
        if key in seen_keys or len(chunk["text"]) < 40:
            continue
        seen_keys.add(key)
        candidates.append(chunk)

    if not candidates:
        return []

    # ── Step 4: Rerank with cosine similarity + keyword boost ─────────────────
    candidate_texts = [c["text"] for c in candidates]
    candidate_embeddings = generate_embeddings(candidate_texts)

    scored = []
    for idx, emb in enumerate(candidate_embeddings):
        cosine_score = float(np.dot(combined_query, emb))
        boosted_score = _keyword_boost(query, candidates[idx]["text"], cosine_score)
        scored.append((boosted_score, idx))

    scored.sort(reverse=True, key=lambda x: x[0])

    # ── Step 5: Score threshold — always return at least top 2 ───────────────
    MIN_SCORE = 0.25
    top_k = scored[:k]
    above = [(s, i) for s, i in top_k if s >= MIN_SCORE]
    if not above:
        above = top_k[:2]   # fallback: return best 2 regardless of score

    results = []
    for score, idx in above:
        c = candidates[idx]
        results.append({
            "text":     c["text"],
            "doc_id":   c["doc_id"],
            "filename": c.get("filename", c["doc_id"]),
            "page":     c.get("page", 1),
            "score":    round(score, 4),
        })

    return results
