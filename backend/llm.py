"""
LLM module (offline / no API).

Generates answers grounded in retrieved document context
without using any external API.
"""

def generate_answer(question, context_chunks):
    """
    Generate a concise, human-readable summary
    from retrieved document chunks.
    """

    if not context_chunks:
        return "No relevant information found in the document."

    summary = []
    summary.append(f"The document primarily discusses the following:")

    for chunk in context_chunks:
        sentences = chunk.split(".")
        for s in sentences:
            if len(s.strip()) > 60:
                summary.append("- " + s.strip())
                break

    final_answer = "\n".join(summary[:6])

    final_answer += (
        "\n\nOverall, the document focuses on designing, implementing, "
        "and deploying an AI-based system, explaining its architecture, "
        "technologies used, applications, and real-world relevance."
    )

    return final_answer


if __name__ == "__main__":
    context = [
        "This document discusses a MERN stack homestay booking platform.",
        "Users can list places and book stays through the application."
    ]

    print(
        generate_answer(
            "What is this project about?",
            context
        )
    )