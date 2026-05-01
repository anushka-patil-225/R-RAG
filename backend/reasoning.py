"""
Reasoning engine module.
Implements a multi-step reasoning pipeline:
  1. Detect if question needs cross-document lookup
  2. Retrieve evidence chunks per sub-query
  3. Synthesize a structured grounded answer
  4. Parse the answer into clean fields for the UI
"""

import re
import numpy as np
from backend.retrieve import retrieve_top_k
from backend.llm import call_llm


# ─── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise analytical AI assistant.
Your job is to reason carefully over provided evidence and answer questions accurately.
- Base answers ONLY on provided context/evidence
- Never hallucinate or invent facts
- Be clear, structured, and concise"""

SYNTHESIS_SYSTEM_PROMPT = """You are a precise AI assistant that answers questions strictly from provided evidence.

RULES:
- Use ONLY the facts explicitly stated in the evidence. Do not invent or guess anything.
- If the answer requires connecting facts across multiple sources (e.g. find a person's project, then find that project's domain), follow that chain step by step.
- If a fact is not in the evidence, say "Not found in the provided documents."
- Never confuse different entities (e.g. different employees, different projects).

OUTPUT FORMAT — you MUST use exactly these labeled sections, each on its own line:
ANSWER: <1-2 sentence direct answer to the question>
EVIDENCE:
- <one specific fact used, with source filename>
- <one specific fact used, with source filename>
REASONING: <one sentence explaining the logical steps you followed>

Do not add any text before ANSWER: or after REASONING:."""


# ─── STEP 1: Decompose question ────────────────────────────────────────────────

def decompose_question(question):
    """
    Decide if the question needs multiple retrieval passes.
    Short questions skip LLM decomposition entirely.
    """
    if len(question.split()) < 12:
        return [question]

    prompt = f"""Analyze this question and decide if it needs to be answered in multiple lookup steps.

Example requiring multiple steps: "What is the deadline of the project that Priya works on?"
  → Step 1: Which project does Priya work on?
  → Step 2: What is the deadline of that project?

Example NOT requiring multiple steps: "What is Project Atlas's domain?"
  → Step 1: What is Project Atlas's domain? (done in one lookup)

Question: {question}

If single-step, return exactly:
1. {question}

If multi-step, return a numbered list of 2-3 focused sub-questions.
Return ONLY the numbered list. Nothing else."""

    result = call_llm(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=250)

    if not result or "❌" in result:
        return [question]

    sub_queries = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^[\d\.\)\-\•\*]+\s*', '', line).strip()
        if cleaned and len(cleaned) > 8:
            sub_queries.append(cleaned)

    return sub_queries if sub_queries else [question]


# ─── STEP 2: Synthesize from raw evidence ──────────────────────────────────────

def synthesize_answer(original_question, evidence_block: str):
    """Call the LLM to produce a labeled structured answer."""
    prompt = f"""Question: {original_question}

EVIDENCE FROM DOCUMENTS:
{evidence_block}

Now answer following the output format in your instructions exactly."""

    return call_llm(prompt, system_prompt=SYNTHESIS_SYSTEM_PROMPT, max_tokens=800)


# ─── STEP 3: Parse LLM output into clean fields ───────────────────────────────

def parse_structured_answer(raw: str):
    """
    Parse the LLM's labeled output into separate fields.
    Returns dict: { answer_text, evidence_points, reasoning_text }
    Falls back gracefully if the model didn't follow the format.
    """
    answer_text = ""
    evidence_points = []
    reasoning_text = ""

    if not raw:
        return {"answer_text": "No answer generated.", "evidence_points": [], "reasoning_text": ""}

    # Extract ANSWER:
    answer_match = re.search(r'ANSWER:\s*(.+?)(?=EVIDENCE:|REASONING:|$)', raw, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()

    # Extract EVIDENCE: bullet lines
    evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?=REASONING:|$)', raw, re.IGNORECASE | re.DOTALL)
    if evidence_match:
        evidence_block = evidence_match.group(1).strip()
        for line in evidence_block.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading bullet characters
            cleaned = re.sub(r'^[\-\•\*\d\.\)]+\s*', '', line).strip()
            if cleaned:
                evidence_points.append(cleaned)

    # Extract REASONING:
    reasoning_match = re.search(r'REASONING:\s*(.+?)$', raw, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()

    # Fallback: if parsing failed, return the raw text as the answer
    if not answer_text and not evidence_points:
        answer_text = raw.strip()

    return {
        "answer_text": answer_text,
        "evidence_points": evidence_points,
        "reasoning_text": reasoning_text,
    }


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def reasoning_pipeline(question, index, documents):
    """
    Full R-RAG reasoning pipeline.
    Returns dict with: question, answer, answer_parsed, reasoning_trace, sub_queries, sources
    """
    reasoning_trace = []

    # Step 1: Decompose
    reasoning_trace.append({
        "step": "Question Analysis",
        "detail": "Determining if the question requires single or multi-step retrieval."
    })
    sub_queries = decompose_question(question)
    reasoning_trace.append({
        "step": "Retrieval Plan",
        "detail": "\n".join(f"• {sq}" for sq in sub_queries)
    })

    # Step 2: Retrieve and build evidence block
    evidence_block = ""
    all_sources = []
    seen_chunks = set()
    all_scores = []

    for sq in sub_queries:
        reasoning_trace.append({
            "step": f"Searching: '{sq}'",
            "detail": "Running semantic search and reranking..."
        })
        chunks = retrieve_top_k(sq, index, documents, k=5)

        if not chunks:
            evidence_block += f"\n[Query: {sq}]\nNo relevant documents found.\n\n"
            continue

        evidence_block += f"\n[Query: {sq}]\n"
        for chunk in chunks:
            chunk_key = chunk["text"][:200]
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)
            evidence_block += (
                f"[Source: {chunk['filename']} | Page {chunk['page']} | Relevance: {chunk['score']}]\n"
                f"{chunk['text']}\n\n"
            )
            all_scores.append(chunk["score"])
            all_sources.append(chunk)

    reasoning_trace.append({
        "step": "Evidence Collected",
        "detail": f"{len(seen_chunks)} unique chunks from {len(sub_queries)} retrieval pass(es)."
    })

    # Step 3: Low-confidence check
    LOW_CONFIDENCE_THRESHOLD = 0.30
    low_confidence = False
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        if avg_score < LOW_CONFIDENCE_THRESHOLD:
            low_confidence = True
            reasoning_trace.append({
                "step": "⚠️ Low Confidence Warning",
                "detail": f"Average relevance score: {avg_score:.2f} (below threshold {LOW_CONFIDENCE_THRESHOLD}). Answer may be imprecise."
            })

    # Step 4: Synthesize
    reasoning_trace.append({
        "step": "Synthesizing Answer",
        "detail": "Generating grounded answer from collected evidence..."
    })
    raw_answer = synthesize_answer(question, evidence_block)

    if not raw_answer or "❌" in (raw_answer or ""):
        raw_answer = "ANSWER: Unable to generate an answer. Please check your LLM connection.\nEVIDENCE:\nREASONING:"

    # Step 5: Parse into structured fields
    parsed = parse_structured_answer(raw_answer)

    if low_confidence:
        parsed["low_confidence"] = True

    # Deduplicate sources
    unique_sources = []
    seen = set()
    for s in all_sources:
        key = s["text"][:200]
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "question": question,
        "answer": raw_answer,           # kept for backward compat
        "answer_parsed": parsed,        # structured fields for UI
        "sub_queries": sub_queries,
        "reasoning_trace": reasoning_trace,
        "sources": unique_sources[:8]
    }
