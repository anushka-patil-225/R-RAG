"""
Retrieval module.

Performs semantic search over stored document embeddings.
"""

import numpy as np
from backend.embed import generate_embeddings
from backend.store import load_index

def retrieve_top_k(query, index, documents, k=5):
    query_embedding = generate_embeddings([query])
    distances, indices = index.search(query_embedding.astype("float32"), k * 2)

    results = []
    for i in indices[0]:
        chunk = documents[i]

        # ❌ Filter low-quality chunks
        if "http" in chunk.lower():
            continue
        if "references" in chunk.lower():
            continue
        if "available at" in chunk.lower():
            continue

        results.append(chunk)

        if len(results) == k:
            break

    return results

if __name__ == "__main__":
    from backend.ingest import load_documents, chunk_text
    from backend.store import create_index, add_embeddings

    docs = load_documents("data/uploaded_docs")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    embeddings = generate_embeddings(chunks)

    index = create_index(embeddings.shape[1])
    add_embeddings(index, embeddings)

    results = retrieve_top_k(
        query="What is this document about?",
        index=index,
        documents=chunks,
        k=3
    )

    print("Top results:")
    for r in results:
        print("-", r[:100])