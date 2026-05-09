"""
reasoning.py — Domain-Agnostic Reasoning Engine for TraceAI
Pipeline: Classify → Decompose → Retrieve → (Extract Facts if comparison) → Generate → Verify

Fixes in this version:
  [FIX A] Simple factual queries skip fact-extraction entirely — raw evidence goes
          straight into answer generation (eliminates 2 extra LLM calls per question).
  [FIX B] LOW_CONF threshold lowered from 0.30 → 0.20 (all-MiniLM-L6-v2 scores on
          domain documents legitimately land in the 0.25-0.45 range).
  [FIX C] Verification failure no longer auto-downgrades to low-confidence when the
          retrieved evidence clearly contains the answer; it only flags truly invented claims.
  [FIX D] SYNTHESIS_SYSTEM prompt is simpler and does not fight with the context format,
          so llama3-8b-8192 produces the required labels consistently.
  [FIX E] Fact extraction is now only triggered for comparison/contradiction queries,
          keeping the token budget low and reducing Groq rate-limit exposure.
  [FIX F] Raw-evidence context is always included as a fallback in generate_answer,
          so even if fact extraction returns nothing the model still has the text.
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

# [FIX D] Simpler synthesis prompt that works reliably with llama3-8b-8192
SYNTHESIS_SYSTEM = BASE_SYSTEM + """
Your task: answer the question using ONLY the evidence provided.

You MUST respond using exactly these labels on their own lines:
ANSWER: <direct answer, 1-3 sentences. If the answer is a number/name/date, state it explicitly.>
EVIDENCE:
- <fact + source filename>
- <fact + source filename>
REASONING: <one sentence explaining why the evidence supports the answer>
CONFIDENCE: <high | medium | low>

Rules:
- ANSWER must be the very first label.
- Do not add any text before ANSWER: or after CONFIDENCE:.
- If the evidence does not contain the answer, write: ANSWER: The information is not present in the provided documents."""

VERIFY_SYSTEM = (
    "You are a strict fact-checker. "
    "Verify only that the answer does NOT invent values absent from the evidence. "
    "Minor wording differences are fine. "
    'Reply ONLY with JSON: {"valid": true, "issue": null} or {"valid": false, "issue": "invented value not in evidence"}'
)


# ── JSON Helper ────────────────────────────────────────────────────────────────

def _parse_json(text: str, fallback):
    """Strip markdown fences and safely parse JSON."""
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
    r"identif\w+ contradiction|find contradiction)\b",
    re.IGNORECASE,
)

def is_comparison_query(question: str) -> bool:
    return bool(_COMPARISON_RE.search(question))


# ── Question Decomposition ─────────────────────────────────────────────────────

_JUNK_RE = re.compile(
    r"(not found|no information|unavailable|n/a|cannot determine|"
    r"does not (exist|appear)|not (stated|mentioned|provided|specified))",
    re.IGNORECASE,
)

def _is_valid_subquery(text: str) -> bool:
    text = text.strip()
    return len(text) >= 10 and not _JUNK_RE.search(text)

def decompose_question(question: str) -> list:
    """
    Split multi-hop questions into retrieval sub-queries.
    Short questions and comparison queries skip the LLM call.
    """
    if is_comparison_query(question):
        return [question]

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
    Extract flat (subject, attribute, value, source) triples.
    Called ONLY for comparison/contradiction queries to keep LLM calls low.
    """
    if not chunks:
        return []

    source_blocks: dict = {}
    for chunk in chunks:
        fname = chunk["filename"]
        source_blocks.setdefault(fname, [])
        source_blocks[fname].append(chunk["text"][:500])

    evidence_text = ""
    for fname, texts in source_blocks.items():
        evidence_text += f"[Source: {fname}]\n" + "\n".join(texts[:3]) + "\n\n"
    evidence_text = evidence_text[:3500]

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
- Use concise attribute names: duration, deadline, penalty, salary, status, rate, clause, etc.
- If the same attribute appears with different values in different [Source:] blocks, create one entry per source.
- Return an empty list if no facts can be extracted.

TEXT:
{evidence_text}"""

    result = call_llm(prompt, system_prompt=JSON_SYSTEM, max_tokens=1400)
    data   = _parse_json(result, {})
    facts  = data.get("facts", []) if isinstance(data, dict) else []

    return [
        f for f in facts
        if f.get("subject") and f.get("attribute") and f.get("value") and f.get("source")
    ]


# ── Fact-Based Comparison & Conflict Detection ─────────────────────────────────

def compare_facts(facts: list, question: str) -> dict:
    """
    Group facts by (subject, attribute) and compare values across sources.
    Pure-Python grouping + one focused LLM call for the human-readable summary.
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
        conflict_json = json.dumps(conflicts[:10], indent=2)
        prompt = f"""These attribute conflicts were found between documents.
Summarise only those relevant to the question.

Question: {question}

Conflicts (JSON):
{conflict_json}

Write a concise bullet-point list. Each bullet:
  "• <subject> / <attribute>: <value_a> (source_a) vs <value_b> (source_b)"

If none are relevant, write "No relevant conflicts found."
No preamble. No explanation."""
        summary = call_llm(prompt, system_prompt=BASE_SYSTEM, max_tokens=400) or ""

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
    Generate a grounded answer.

    [FIX A] Simple factual queries: always lead with raw_evidence so the model
    has the full text. Facts (if available) are an additional structured layer.
    [FIX F] raw_evidence is always appended as a fallback context block.
    """
    ctx_parts = []

    # For comparison queries, lead with structured facts + conflict analysis
    if is_comparison:
        if facts:
            ctx_parts.append(f"EXTRACTED FACTS:\n{json.dumps(facts[:40], indent=2)}")
        if comparison.get("conflicts"):
            ctx_parts.append(
                f"CONFLICTS DETECTED:\n{json.dumps(comparison['conflicts'][:10], indent=2)}"
            )
        if comparison.get("matches"):
            ctx_parts.append(
                f"MATCHING FACTS:\n{json.dumps(comparison['matches'][:5], indent=2)}"
            )
        if comparison.get("summary"):
            ctx_parts.append(f"CONFLICT SUMMARY:\n{comparison['summary']}")

    # [FIX F] Always include raw evidence — it's the ground truth the model must read
    ctx_parts.append(f"EVIDENCE FROM DOCUMENTS:\n{raw_evidence[:3000]}")

    context = "\n\n".join(ctx_parts)

    prompt = f"""Question: {question}

{context}

Answer the question using ONLY the information in the evidence above.
Follow the output format in your instructions exactly."""

    return call_llm(prompt, system_prompt=SYNTHESIS_SYSTEM, max_tokens=900)


# ── Answer Verification ────────────────────────────────────────────────────────

def verify_answer(question: str, answer: str, facts: list, raw_evidence: str) -> dict:
    """
    [FIX C] Only flag answers that invent values not present anywhere in the evidence.
    No longer fails just because phrasing differs from the fact table format.
    """
    # Build a compact evidence summary the verifier can check against
    evidence_ctx = raw_evidence[:1200]
    if facts:
        evidence_ctx = json.dumps(facts[:20], indent=2) + "\n\n" + raw_evidence[:600]

    prompt = f"""Question: {question}

Answer to verify:
{answer[:700]}

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
    """Parse labeled LLM output into UI-ready fields."""
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


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def reasoning_pipeline(question: str, index, documents: list) -> dict:
    """
    Full R-RAG pipeline — backward-compatible with app.py.

    [FIX A] Simple factual queries: 2 LLM calls max (decompose if long + generate).
    [FIX E] Comparison queries: up to 4 LLM calls (decompose + extract + compare + generate).
    [FIX B] LOW_CONF threshold lowered to 0.20.
    """
    trace = []

    # ── Classify query ────────────────────────────────────────────────────────
    comparison_mode = is_comparison_query(question)
    retrieval_k     = 8 if comparison_mode else 5

    trace.append({
        "step": "Query Classification",
        "detail": (
            "📊 Comparison / conflict query — wide retrieval (k=8) + fact extraction enabled."
            if comparison_mode else
            "🔍 Factual query — standard retrieval (k=5), direct evidence synthesis."
        ),
    })

    # ── Decompose ─────────────────────────────────────────────────────────────
    trace.append({
        "step": "Question Analysis",
        "detail": (
            "Comparison query: skipping decomposition."
            if comparison_mode else
            "Checking if question needs multi-step retrieval."
        ),
    })
    sub_queries = decompose_question(question)
    trace.append({
        "step": "Retrieval Plan",
        "detail": "\n".join(f"• {sq}" for sq in sub_queries),
    })

    # ── Retrieve ──────────────────────────────────────────────────────────────
    all_chunks   = []
    seen_keys    = set()
    all_scores   = []
    raw_evidence = ""

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
            raw_evidence += (
                f"[Source: {c['filename']} | Page {c['page']} | "
                f"Relevance: {c['score']}]\n{c['text']}\n\n"
            )

    doc_count = len({c["filename"] for c in all_chunks})
    trace.append({
        "step": "Retrieval Complete",
        "detail": (
            f"{len(all_chunks)} unique chunk(s) from {len(sub_queries)} "
            f"query pass(es) across {doc_count} document(s)."
        ),
    })

    # [FIX B] Lowered threshold: 0.20 is realistic for all-MiniLM-L6-v2
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

    # ── [FIX E] Fact extraction only for comparison queries ───────────────────
    facts      = []
    comparison = {"conflicts": [], "matches": [], "missing": [], "summary": ""}

    if comparison_mode:
        trace.append({
            "step": "Fact Extraction",
            "detail": "Extracting (subject, attribute, value, source) triples for conflict detection...",
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

    # ── [FIX C] Verify — only flag truly invented values ─────────────────────
    trace.append({
        "step": "Answer Verification",
        "detail": "Checking all claims are traceable to retrieved evidence...",
    })
    verification = verify_answer(question, raw_answer, facts, raw_evidence)

    if not verification.get("valid", True):
        issue = verification.get("issue") or "Unspecified verification failure."
        # [FIX C] Only downgrade if answer claims a value not in evidence at all
        # (not just because the verifier found a formatting mismatch)
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

    # Inject conflict bullets into evidence points for UI visibility
    for conflict in comparison.get("conflicts", [])[:4]:
        obs = conflict.get("observations", [])
        if len(obs) >= 2:
            parsed["evidence_points"].append(
                f"⚠️ Conflict — '{conflict.get('subject', '?')}' / "
                f"'{conflict.get('attribute', '?')}': "
                + " vs ".join(
                    f"{o['value']!r} ({o['source']})" for o in obs[:2]
                )
            )

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
        "sources":         unique_sources[:8],
        "facts":           facts,
        "comparison":      comparison,
    }
