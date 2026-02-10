"""
Reasoning module.

Implements reasoning-aware Retrieval-Augmented Generation (R-RAG):
- Accepts a user question
- Retrieves relevant document chunks
- Synthesizes a grounded answer
- Returns answer with traceable sources
"""

from backend.retrieve import retrieve_top_k
from backend.llm import generate_answer


def reasoning_pipeline(question, index, documents):
    """
    End-to-end reasoning-aware RAG pipeline.

    Args:
        question (str): User question
        index (faiss.Index): FAISS vector index
        documents (list): List of document chunks

    Returns:
        dict: Contains question, answer, and sources
    """

    # Step 1: Retrieve relevant document chunks
    context_chunks = retrieve_top_k(
        query=question,
        index=index,
        documents=documents,
        k=3
    )

    # Step 2: Generate a grounded answer using retrieved context
    answer = generate_answer(
        question=question,
        context_chunks=context_chunks
    )

    # Step 3: Return structured, explainable output
    return {
        "question": question,
        "answer": answer,
        "sources": context_chunks
    }


# -------------------- TEST BLOCK --------------------

if __name__ == "__main__":
    from backend.ingest import load_documents, chunk_text
    from backend.embed import generate_embeddings
    from backend.store import create_index, add_embeddings

    # Load documents
    docs = load_documents("data/uploaded_docs")

    # Chunk documents
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Create FAISS index
    index = create_index(embeddings.shape[1])
    add_embeddings(index, embeddings)

    # Run reasoning pipeline
    result = reasoning_pipeline(
        question="What is this document about?",
        index=index,
        documents=chunks
    )

    # Display output
    print("\nQUESTION:\n", result["question"])
    print("\nFINAL ANSWER:\n", result["answer"])
    print("\nSOURCES:\n")
    for src in result["sources"]:
        print("-", src[:120])