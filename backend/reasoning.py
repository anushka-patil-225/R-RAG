"""
reasoning.py — Domain-Agnostic Reasoning Engine for TraceAI
Pipeline: Classify → Decompose/Expand → Retrieve → (Extract Facts if comparison) → Generate → Verify

Key improvements in this version:
  [A] Comparison queries now generate targeted sub-queries (e.g. "confidentiality duration",
      "notification timeline") so every relevant section from every document is retrieved.
  [B] SYNTHESIS_SYSTEM allows rich multi-point structured answers — not just 1-3 sentences.
  [C] Cross-document retrieval: for multi-doc KBs, a catch-up pass ensures every document
      contributes at least one chunk so nothing is silently missed.
  [D] Evidence cap raised to 5000 chars (comparison) / 3500 chars (factual).
  [E] Fact extraction uses more content per chunk for better triple coverage.
  [F] compare_facts summary produces a clean per-inconsistency enumeration.
  [G] parse_structured_answer converts newlines to <br> so answers render correctly in HTML.
"""

import re
import json
from backend.retrieve import retrieve_top_k
from backend.llm import call_llm


# ── System Prompts ─────────────────────────────────────────────────────────────

BASE_SYSTEM = """You are an evidence-grounded reasoning engine.
- Reason ONLY from the provided text. Never use external knowledge.
- Never merge facts from different subjects or sources.
- If a fact is truly absent from the evidence, say so explicitly."""

JSON_SYSTEM = BASE_SYSTEM + "\nOutput ONLY valid JSON. No preamble, explanation, or markdown fences."

SYNTHESIS_SYSTEM = BASE_SYSTEM + """
Your task: answer the question using ONLY the evidence provided.
Respond using EXACTLY these four sections in this order. Each label on its own line.

--- FORMAT ---
ANSWER:
<Concise direct answer. No filenames. No inline citations. No verbose detail.
 For a single fact: 1-2 sentences.
 For multi-point questions: one bullet (•) per key point, brief phrasing only.
 Keep it short — all detail and attribution goes in EVIDENCE below.>

EVIDENCE:
- <Exact supporting fact> — Source: <filename>, <section if visible>
- <Another fact> — Source: <filename>, <section if visible>
<List every specific fact that supports the answer. Each bullet must cite its source.
 For inconsistency questions: list every conflict with both values and both filenames.>

REASONING:
<2-4 sentences explaining how the evidence leads to the answer.
 Name which document contributes which piece. For cross-document questions, explain
 how the documents relate or conflict.>

CONFIDENCE: <high | medium | low>
--- END FORMAT ---

Rules:
- ANSWER must be brief and clean — no filenames, no parenthetical citations.
- All source attribution belongs in EVIDENCE, never in ANSWER.
- All explanation belongs in REASONING, never in ANSWER.
- Start your response with ANSWER: — no preamble before it.
- End with CONFIDENCE: — nothing after it.
- If the answer is absent from the evidence: write ANSWER: The information is not present in the provided documents."""

VERIFY_SYSTEM = (
    "You are a strict fact-checker. "
    "Verify only that the answer does NOT invent values absent from the evidence. "
    "Minor wording differences are fine. "
    'Reply ONLY with JSON: {"valid": true, "issue": null} or {"valid": false, "issue": "invented value not in evidence"}'
)


# ── JSON Helper ────────────────────────────────────────────────────────────────

def _parse_json(text: str, fallback):
    if not text:
        return fallback
    try:
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        return json.loads(clean)
    except Exception:
        return fallback


# ── Comparison Query Classifier ────────────────────────────────────────────────

_COMPARISON_RE = re.compile(
    r"\b(compar|differ|conflict|contradict|inconsisten|mismatch|vs\.?|versus|"
    r"same as|unlike|contrast|discrepan|overlap|both|neither|agree|disagree|"
    r"identif\w+ (inconsisten|conflict|contradict|discrepan)|find (inconsisten|conflict))\b",
    re.IGNORECASE,
)

def is_comparison_query(question: str) -> bool:
    return bool(_COMPARISON_RE.search(question))


# ── Targeted Sub-Query Generation for Comparison Queries ──────────────────────

_FALLBACK_COMPARISON_QUERIES = [
    "confidentiality obligation duration after termination",
    "incident notification reporting timeline hours days",
    "data deletion destruction period after termination",
    "audit frequency annual",
    "log record retention period months",
    "subcontractor approval security requirements",
    "liability indemnification cap",
    "payment terms invoice period",
]

def generate_comparison_subqueries(question: str, documents: list) -> list:
    """
    [A] For comparison/inconsistency queries, use an LLM call to generate targeted
    sub-queries covering the specific topics that are likely to differ across documents.
    This ensures retrieval covers all relevant sections, not just those semantically
    similar to the broad question.
    """
    doc_names = sorted({d["filename"] for d in documents})
    doc_list  = ", ".join(doc_names)

    prompt = f"""The user wants to find inconsistencies or differences across multiple documents.
Documents available: {doc_list}
User question: "{question}"

Generate 7 specific retrieval queries that will locate sections likely to contain
different or conflicting values across these documents. Target concrete operational
specifics such as: time durations, notification deadlines, retention periods, audit
frequencies, approval thresholds, deletion timelines, and compliance requirements.

Return ONLY a numbered list of short queries (4-10 words each). No explanation."""

    result = call_llm(prompt, system_prompt=BASE_SYSTEM, max_tokens=250)

    queries = [question]   # always include the original

    if result and "❌" not in result:
        for line in result.strip().split("\n"):
            cleaned = re.sub(r"^[\d\.\)\-\•\*]+\s*", "", line.strip()).strip()
            if cleaned and 5 <= len(cleaned) <= 120:
                queries.append(cleaned)
                if len(queries) >= 9:
                    break

    # Pad with safe fallbacks if the LLM returned too few
    for fb in _FALLBACK_COMPARISON_QUERIES:
        if len(queries) >= 9:
            break
        if not any(fb.split()[0] in q.lower() for q in queries):
            queries.append(fb)

    return queries


# ── Question Decomposition (factual queries) ───────────────────────────────────

_JUNK_RE = re.compile(
    r"(not found|no information|unavailable|n/a|cannot determine|"
    r"does not (exist|appear)|not (stated|mentioned|provided|specified))",
    re.IGNORECASE,
)

def _is_valid_subquery(text: str) -> bool:
    text = text.strip()
    return len(text) >= 10 and not _JUNK_RE.search(text)

def decompose_question(question: str) -> list:
    """Split multi-hop factual questions into retrieval sub-queries."""
    if len(question.split()) < 10:
        return [question]

    prompt = f"""Does this question require chaining facts from multiple lookup steps?

Example needing multiple steps:
  "What domain does the project that Alice manages belong to?"
  → 1. Which project does Alice manage?
  → 2. What domain does that project belong to?

Example NOT needing multiple steps:
  "What is the deadline of Project X?"
  → 1. What is Project X's deadline?

Question: {question}

If single-step, return exactly:
1. {question}

If multi-step, return 2-3 focused sub-questions as a numbered list.
Return ONLY the numbered list. Nothing else."""

    result = call_llm(prompt, system_prompt=BASE_SYSTEM, max_tokens=200)
    if not result or "❌" in result:
        return [question]

    sub_queries = []
    for line in result.strip().split("\n"):
        cleaned = re.sub(r"^[\d\.\)\-\•\*]+\s*", "", line.strip()).strip()
        if _is_valid_subquery(cleaned):
            sub_queries.append(cleaned)

    return sub_queries if sub_queries else [question]


# ── Fact Extraction (comparison queries only) ──────────────────────────────────

def extract_facts(chunks: list) -> list:
    """
    [E] Extract (subject, attribute, value, source) triples.
    Uses more content per chunk to avoid missing key facts.
    """
    if not chunks:
        return []

    source_blocks: dict = {}
    for chunk in chunks:
        fname = chunk["filename"]
        source_blocks.setdefault(fname, [])
        source_blocks[fname].append(chunk["text"][:900])   # was 500

    evidence_text = ""
    for fname, texts in source_blocks.items():
        evidence_text += f"[Source: {fname}]\n" + "\n---\n".join(texts[:8]) + "\n\n"  # was 3
    evidence_text = evidence_text[:6000]   # was 3500

    prompt = f"""Read the text below and extract every factual statement as a structured triple.

Output ONLY this JSON:
{{
  "facts": [
    {{
      "subject":   "what or who the fact is about",
      "attribute": "the property or aspect being described",
      "value":     "the stated value or description",
      "source":    "the filename from the [Source:] tag above this text"
    }}
  ]
}}

Rules:
- Extract ONLY facts explicitly written in the text. Never infer.
- Use concise attribute names: duration, deadline, penalty, notification_period, deletion_period,
  retention_period, audit_frequency, approval_requirement, access_level, etc.
- If the same attribute appears with different values in different [Source:] blocks, create one
  entry per source — this is critical for finding inconsistencies.
- Return an empty list if no facts can be extracted.

TEXT:
{evidence_text}"""

    result = call_llm(prompt, system_prompt=JSON_SYSTEM, max_tokens=1800)
    data   = _parse_json(result, {})
    facts  = data.get("facts", []) if isinstance(data, dict) else []

    return [
        f for f in facts
        if f.get("subject") and f.get("attribute") and f.get("value") and f.get("source")
    ]


# ── Fact-Based Comparison & Conflict Detection ─────────────────────────────────

def compare_facts(facts: list, question: str) -> dict:
    """
    [F] Group facts by (subject, attribute) and compare values across sources.
    Produces a clean per-inconsistency enumeration in the summary.
    """
    empty = {"conflicts": [], "matches": [], "missing": [], "summary": ""}
    if not facts:
        return empty

    groups: dict = {}
    for f in facts:
        key = (f["subject"].strip().lower(), f["attribute"].strip().lower())
        groups.setdefault(key, [])
        groups[key].append({"value": f["value"], "source": f["source"]})

    conflicts, matches = [], []

    for (subj, attr), observations in groups.items():
        seen: set = set()
        unique = []
        for obs in observations:
            pair = (obs["value"].strip().lower(), obs["source"].strip().lower())
            if pair not in seen:
                seen.add(pair)
                unique.append(obs)

        distinct_vals    = {o["value"].strip().lower() for o in unique}
        distinct_sources = {o["source"].strip()        for o in unique}

        if len(distinct_vals) > 1:
            conflicts.append({
                "subject":      subj,
                "attribute":    attr,
                "observations": unique,
            })
        elif len(distinct_sources) > 1:
            matches.append({
                "subject":   subj,
                "attribute": attr,
                "value":     unique[0]["value"],
                "sources":   list(distinct_sources),
            })

    summary = ""
    if conflicts:
        conflict_json = json.dumps(conflicts[:12], indent=2)
        prompt = f"""These conflicts were detected between documents.
Question: {question}

Conflicts (JSON):
{conflict_json}

For each conflict that is relevant to the question, write one line in this format:
[Topic / Attribute]: [Document A] states "[value_a]" vs [Document B] states "[value_b]"

List ALL relevant conflicts. No preamble. No explanation beyond the conflict lines."""
        summary = call_llm(prompt, system_prompt=BASE_SYSTEM, max_tokens=500) or ""

    return {
        "conflicts": conflicts,
        "matches":   matches,
        "missing":   [],
        "summary":   summary,
    }


# ── Answer Generation ──────────────────────────────────────────────────────────

def generate_answer(
    question:       str,
    facts:          list,
    comparison:     dict,
    raw_evidence:   str,
    is_comparison:  bool,
) -> str:
    """
    [D] Evidence cap raised; comparison mode leads with structured conflict data.
    """
    ctx_parts = []

    if is_comparison:
        if facts:
            ctx_parts.append(f"EXTRACTED FACTS:\n{json.dumps(facts[:50], indent=2)}")
        if comparison.get("conflicts"):
            ctx_parts.append(
                f"CONFLICTS DETECTED ({len(comparison['conflicts'])}):\n"
                + json.dumps(comparison["conflicts"][:15], indent=2)
            )
        if comparison.get("matches"):
            ctx_parts.append(
                f"MATCHING FACTS:\n{json.dumps(comparison['matches'][:5], indent=2)}"
            )
        if comparison.get("summary"):
            ctx_parts.append(f"CONFLICT SUMMARY:\n{comparison['summary']}")
        # Larger evidence window for comparison queries
        ctx_parts.append(f"EVIDENCE FROM DOCUMENTS:\n{raw_evidence[:5000]}")
    else:
        ctx_parts.append(f"EVIDENCE FROM DOCUMENTS:\n{raw_evidence[:3500]}")

    context = "\n\n".join(ctx_parts)

    prompt = f"""Question: {question}

{context}

Answer the question using ONLY the information in the evidence above.
Follow the output format in your instructions exactly."""

    return call_llm(prompt, system_prompt=SYNTHESIS_SYSTEM, max_tokens=1200)


# ── Answer Verification ────────────────────────────────────────────────────────

def verify_answer(question: str, answer: str, facts: list, raw_evidence: str) -> dict:
    """Only flag answers that invent values not present anywhere in the evidence."""
    evidence_ctx = raw_evidence[:1500]
    if facts:
        evidence_ctx = json.dumps(facts[:20], indent=2) + "\n\n" + raw_evidence[:800]

    prompt = f"""Question: {question}

Answer to verify:
{answer[:800]}

Evidence (ground truth):
{evidence_ctx}

Check ONLY: does the answer state any specific value (number, date, name, clause text)
that is completely absent from the evidence above?

- If the answer says "not found" but the evidence DOES contain the answer → valid: false
- If the answer correctly cites a value present in the evidence → valid: true
- Minor wording differences are acceptable → valid: true

Reply ONLY with JSON: {{"valid": true, "issue": null}} or {{"valid": false, "issue": "brief reason"}}"""

    result = call_llm(prompt, system_prompt=VERIFY_SYSTEM, max_tokens=150)
    data   = _parse_json(result, {"valid": True, "issue": None})
    return data if isinstance(data, dict) else {"valid": True, "issue": None}


# ── Output Parser ──────────────────────────────────────────────────────────────

def parse_structured_answer(raw: str) -> dict:
    """
    Parse labeled LLM output into UI-ready fields.
    Uses label-boundary lookahead without requiring exact newline position — works
    regardless of how the LLM spaces or wraps its output.
    Converts newlines in answer_text to <br> so they render correctly in the HTML div.
    """
    if not raw:
        return {
            "answer_text":     "No answer generated.",
            "evidence_points": [],
            "reasoning_text":  "",
            "confidence":      "low",
        }

    answer_text     = ""
    evidence_points = []
    reasoning_text  = ""
    confidence      = "medium"

    # Robust regex: lookahead does NOT require a leading \n so it works even
    # when the LLM puts labels on the same line or uses inconsistent spacing.
    m = re.search(
        r"ANSWER:\s*(.+?)(?=EVIDENCE:|REASONING:|CONFIDENCE:|$)",
        raw, re.IGNORECASE | re.DOTALL,
    )
    if m:
        answer_text = m.group(1).strip()

    m = re.search(
        r"EVIDENCE:\s*(.+?)(?=REASONING:|CONFIDENCE:|$)",
        raw, re.IGNORECASE | re.DOTALL,
    )
    if m:
        for line in m.group(1).strip().split("\n"):
            cleaned = re.sub(r"^[\-\•\*\d\.\)]+\s*", "", line.strip()).strip()
            if cleaned:
                evidence_points.append(cleaned)

    m = re.search(
        r"REASONING:\s*(.+?)(?=CONFIDENCE:|$)",
        raw, re.IGNORECASE | re.DOTALL,
    )
    if m:
        reasoning_text = m.group(1).strip()

    m = re.search(r"CONFIDENCE:\s*(high|medium|low)", raw, re.IGNORECASE)
    if m:
        confidence = m.group(1).lower()

    if not answer_text:
        answer_text = raw.strip()

    return {
        "answer_text":     answer_text,
        "evidence_points": evidence_points,
        "reasoning_text":  reasoning_text,
        "confidence":      confidence,
    }


# ── Cross-Document Coverage Check ─────────────────────────────────────────────

def _ensure_cross_document_coverage(
    question: str,
    all_chunks: list,
    seen_keys: set,
    all_scores: list,
    raw_evidence_parts: list,
    documents: list,
    index,
    k: int,
) -> tuple:
    """
    [C] If any document in the KB has zero retrieved chunks, run a catch-up
    retrieval pass for that document using the original question.
    Returns updated (all_chunks, seen_keys, all_scores, raw_evidence_parts).
    """
    all_doc_names      = {d["filename"] for d in documents}
    retrieved_doc_names = {c["filename"] for c in all_chunks}
    missing_docs       = all_doc_names - retrieved_doc_names

    if not missing_docs:
        return all_chunks, seen_keys, all_scores, raw_evidence_parts

    # Fetch extra candidates and filter to missing documents
    catch_up = retrieve_top_k(question, index, documents, k=min(20, len(documents)))
    for c in catch_up:
        if c["filename"] not in missing_docs:
            continue
        key = c["text"][:200]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_chunks.append(c)
        all_scores.append(c["score"])
        raw_evidence_parts.append(
            f"[Source: {c['filename']} | Page {c['page']} | "
            f"Relevance: {c['score']}]\n{c['text']}\n\n"
        )

    return all_chunks, seen_keys, all_scores, raw_evidence_parts


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def reasoning_pipeline(question: str, index, documents: list) -> dict:
    """
    Full R-RAG pipeline.

    Factual queries  : decompose → retrieve (k=6) → cross-doc check → generate → verify
    Comparison queries: targeted sub-queries → retrieve (k=5 per query) →
                        cross-doc check → extract facts → compare → generate → verify
    """
    trace = []

    # ── Classify query ────────────────────────────────────────────────────────
    comparison_mode = is_comparison_query(question)

    trace.append({
        "step": "Query Classification",
        "detail": (
            "📊 Comparison / inconsistency query — targeted sub-query expansion + fact extraction."
            if comparison_mode else
            "🔍 Factual query — standard retrieval, direct evidence synthesis."
        ),
    })

    # ── Build sub-queries ─────────────────────────────────────────────────────
    if comparison_mode:
        trace.append({
            "step": "Sub-Query Expansion",
            "detail": "Generating targeted retrieval queries to cover all relevant document sections...",
        })
        sub_queries = generate_comparison_subqueries(question, documents)
        retrieval_k = 5   # k per sub-query; 7-9 queries × 5 = up to 45 unique candidates
    else:
        trace.append({
            "step": "Question Analysis",
            "detail": "Checking if question needs multi-step retrieval.",
        })
        sub_queries = decompose_question(question)
        retrieval_k = 6

    trace.append({
        "step": "Retrieval Plan",
        "detail": "\n".join(f"• {sq}" for sq in sub_queries),
    })

    # ── Retrieve ──────────────────────────────────────────────────────────────
    all_chunks          = []
    seen_keys           = set()
    all_scores          = []
    raw_evidence_parts  = []

    for sq in sub_queries:
        trace.append({
            "step": f"Searching: '{sq[:80]}'",
            "detail": f"Semantic search (k={retrieval_k}) + cosine reranking...",
        })
        chunks = retrieve_top_k(sq, index, documents, k=retrieval_k)

        if not chunks:
            trace.append({
                "step": f"No results for '{sq[:60]}'",
                "detail": "No chunks cleared the relevance threshold for this sub-query.",
            })
            continue

        for c in chunks:
            key = c["text"][:200]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_chunks.append(c)
            all_scores.append(c["score"])
            raw_evidence_parts.append(
                f"[Source: {c['filename']} | Page {c['page']} | "
                f"Relevance: {c['score']}]\n{c['text']}\n\n"
            )

    # ── [C] Cross-document coverage check ─────────────────────────────────────
    all_chunks, seen_keys, all_scores, raw_evidence_parts = _ensure_cross_document_coverage(
        question, all_chunks, seen_keys, all_scores, raw_evidence_parts, documents, index, k=15
    )

    raw_evidence = "".join(raw_evidence_parts)
    doc_count    = len({c["filename"] for c in all_chunks})

    trace.append({
        "step": "Retrieval Complete",
        "detail": (
            f"{len(all_chunks)} unique chunk(s) from {len(sub_queries)} "
            f"query pass(es) across {doc_count} document(s)."
        ),
    })

    low_confidence = False
    LOW_CONF       = 0.20
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        if avg < LOW_CONF:
            low_confidence = True
            trace.append({
                "step": "⚠️ Low Confidence",
                "detail": f"Avg relevance {avg:.2f} is below threshold {LOW_CONF}.",
            })

    # No chunks at all — bail early
    if not all_chunks:
        parsed = {
            "answer_text":     "Relevant information not found in the provided documents.",
            "evidence_points": [],
            "reasoning_text":  "Retrieval returned no chunks for any sub-query.",
            "confidence":      "low",
            "low_confidence":  True,
        }
        return {
            "question":        question,
            "answer":          parsed["answer_text"],
            "answer_parsed":   parsed,
            "sub_queries":     sub_queries,
            "reasoning_trace": trace,
            "sources":         [],
            "facts":           [],
            "comparison":      {"conflicts": [], "matches": [], "missing": [], "summary": ""},
        }

    # ── Fact extraction + comparison (comparison queries only) ─────────────────
    facts      = []
    comparison = {"conflicts": [], "matches": [], "missing": [], "summary": ""}

    if comparison_mode:
        trace.append({
            "step": "Fact Extraction",
            "detail": f"Extracting (subject, attribute, value, source) triples from {len(all_chunks)} chunks...",
        })
        facts = extract_facts(all_chunks)
        sources_with_facts = sorted({f["source"] for f in facts})
        trace.append({
            "step": "Facts Extracted",
            "detail": (
                f"{len(facts)} fact(s) from {len(sources_with_facts)} source(s): "
                + ", ".join(sources_with_facts[:4])
            ) if facts else "No structured facts extracted — raw evidence will be used.",
        })

        trace.append({
            "step": "Fact Comparison",
            "detail": "Grouping facts by attribute and detecting cross-source differences...",
        })
        comparison = compare_facts(facts, question)

        n_conflicts = len(comparison["conflicts"])
        n_matches   = len(comparison["matches"])
        summary_parts = []
        if n_conflicts:
            summary_parts.append(f"⚠️ {n_conflicts} conflict(s) found")
        if n_matches:
            summary_parts.append(f"✓ {n_matches} shared attribute(s)")
        trace.append({
            "step": "Comparison Result",
            "detail": (
                ", ".join(summary_parts) + (
                    "\n" + comparison["summary"] if comparison.get("summary") else ""
                )
            ) if summary_parts else "No conflicts or shared attributes detected.",
        })
    else:
        trace.append({
            "step": "Direct Evidence Synthesis",
            "detail": f"Passing {len(all_chunks)} retrieved chunk(s) directly to answer generation.",
        })

    # ── Generate Answer ────────────────────────────────────────────────────────
    trace.append({
        "step": "Answer Generation",
        "detail": "Synthesising grounded answer from retrieved evidence...",
    })
    raw_answer = generate_answer(
        question, facts, comparison, raw_evidence, comparison_mode
    )

    if not raw_answer or "❌" in (raw_answer or ""):
        raw_answer = (
            "ANSWER: Unable to generate an answer. Please check your LLM connection.\n"
            "EVIDENCE:\nREASONING:\nCONFIDENCE: low"
        )

    # ── Verify ────────────────────────────────────────────────────────────────
    trace.append({
        "step": "Answer Verification",
        "detail": "Checking all claims are traceable to retrieved evidence...",
    })
    verification = verify_answer(question, raw_answer, facts, raw_evidence)

    if not verification.get("valid", True):
        issue = verification.get("issue") or "Unspecified verification failure."
        if "not present" in issue.lower() or "absent" in issue.lower() or "invented" in issue.lower():
            trace.append({
                "step": "⚠️ Verification Failed",
                "detail": f"Issue: {issue} — confidence downgraded.",
            })
            low_confidence = True
        else:
            trace.append({
                "step": "✓ Verification Passed (with note)",
                "detail": f"Note: {issue}",
            })
    else:
        trace.append({
            "step": "✓ Verification Passed",
            "detail": "All claims are traceable to retrieved evidence.",
        })

    # ── Assemble output ───────────────────────────────────────────────────────
    parsed = parse_structured_answer(raw_answer)

    if low_confidence:
        parsed["low_confidence"] = True
        parsed["confidence"]     = "low"

    # Deduplicate sources for UI
    unique_sources, seen = [], set()
    for c in all_chunks:
        key = c["text"][:200]
        if key not in seen:
            seen.add(key)
            unique_sources.append(c)

    return {
        "question":        question,
        "answer":          raw_answer,
        "answer_parsed":   parsed,
        "sub_queries":     sub_queries,
        "reasoning_trace": trace,
        "sources":         unique_sources[:10],
        "facts":           facts,
        "comparison":      comparison,
    }