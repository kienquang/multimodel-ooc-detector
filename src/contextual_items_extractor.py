"""
contextual_items_extractor.py — Stage 1b: Contextual Items → S-V-O Triplets

Implements the paper's QA pipeline (Section III.A) and maps results to
EGMMG S-V-O graph structure.

Paper pipeline:
  For each of 6 context items (People, Location, Date, Event, Object, Motivation):
    1. Combine ranked evidence as context
    2. Ask LLM a targeted question
    3. If evidence is insufficient → return "Unknown" (anti-hallucination)

EGMMG S-V-O mapping:
  People     → Subject node(s) of the claim graph
  Event      → Subject  PERFORMS  Event
  Location   → Subject  LOCATED_IN  Location
  Date       → Subject  OCCURRED_ON  Date
  Motivation → Subject  TARGETS  Motivation
  Object     → Subject  HAS_STATE  Object

The output SVOList is directly comparable against the evidence SVOList
in RetrievalAgent, enabling the conflict detection step.

Note: This replaces the raw-text SVO extraction for evidence.
      Caption SVO extraction (svo_extractor.py) remains unchanged.
"""

from pydantic import BaseModel, Field
from typing import Optional

from src.llm_provider import llm_provider
from src.schemas import SVOList, SVOTriplet


# ──────────────────────────────────────────────────────────────
# INTERMEDIATE: 6 Contextual Items
# ──────────────────────────────────────────────────────────────

class ContextualItems(BaseModel):
    """
    The 6 context attributes defined in the paper.
    Each field is a string answer or "Unknown" if evidence is insufficient.
    """
    people:     str = Field(default="Unknown", description="Who is shown in the image?")
    location:   str = Field(default="Unknown", description="Where was the event taken?")
    date:       str = Field(default="Unknown", description="When was the event taken?")
    event:      str = Field(default="Unknown", description="Which event is depicted?")
    object:     str = Field(default="Unknown", description="Which objects/places are shown?")
    motivation: str = Field(default="Unknown", description="Why was the image taken?")


# ──────────────────────────────────────────────────────────────
# PAPER'S 6 QUESTIONS
# ──────────────────────────────────────────────────────────────

# Exactly as described in paper Section III.A
_CONTEXT_QUESTIONS: dict[str, str] = {
    "people":     "Who is shown in the image?",
    "location":   "Where was the event in this image taken?",
    "date":       "When was the event in this image taken?",
    "event":      "Which event is depicted in the image?",
    "object":     "Which animals, plants, buildings, or objects are shown in the image?",
    "motivation": "Why was the image taken?",
}


# ──────────────────────────────────────────────────────────────
# EVIDENCE PREPARATION
# ──────────────────────────────────────────────────────────────

def _build_evidence_context(
    evidence_list: list[dict],
    visual_entities: list[str],
) -> str:
    """
    Combine ranked evidence into a single context string for the LLM.

    Structure (mirrors paper's evidence input):
      - Visual entities identified by Google Lens (entity-level signal)
      - Webpage titles (often the most informative, concise signals)
      - Article text snippets (longer evidence)
      - Image captions from articles (caption-level signal)
    """
    lines = []

    # Visual entities from Google Lens (highest confidence, from the API itself)
    if visual_entities:
        lines.append("## VISUAL ENTITIES (identified by Google Lens)")
        for ent in visual_entities:
            lines.append(f"  - {ent}")
        lines.append("")

    # Webpage titles + article snippets
    lines.append("## RETRIEVED EVIDENCE")
    for i, ev in enumerate(evidence_list):
        lines.append(f"### Source {i+1} (CLIP score: {ev.get('clip_score', 0):.3f})")
        lines.append(f"Title: {ev.get('title', '')}")

        # Image captions from the article (often very informative for OOC)
        captions = ev.get("image_captions", [])
        if captions:
            lines.append("Image captions found on page:")
            for cap in captions[:3]:
                lines.append(f"  \"{cap}\"")

        lines.append(f"Article text:\n{ev.get('text', '')[:600]}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# STEP 1: QA for 6 context items (paper's prompt template)
# ──────────────────────────────────────────────────────────────

def _extract_contextual_items(
    evidence_context: str,
    source_label: str = "evidence",
) -> ContextualItems:
    """
    Answer all 6 context questions in a single LLM call.

    Paper: Uses Llama 3.1 8B with a prompt template.
    Anti-hallucination: "If the textual evidence does not contain enough
    information to answer the question, return 'Unknown'."

    We answer all 6 in one call (more efficient than 6 separate calls).
    """
    print(f"[Contextual] Extracting 6 context items from {source_label}...")

    questions_block = "\n".join(
        f"  {i+1}. {key.upper()}: {q}"
        for i, (key, q) in enumerate(_CONTEXT_QUESTIONS.items())
    )

    system_prompt = (
        "You are a precise context extractor for misinformation detection. "
        "Answer each question using ONLY the provided evidence. "
        "If the evidence does not contain enough information to answer a question, "
        "you MUST return exactly 'Unknown' for that field. "
        "Never guess or infer beyond what is explicitly stated in the evidence. "
        "Respond in English only."
    )

    user_prompt = f"""Using the evidence below, answer these 6 questions about the image:

{questions_block}

---
EVIDENCE:
{evidence_context}
---

Return ONLY a JSON object with these exact keys:
{{
  "people": "...",
  "location": "...",
  "date": "...",
  "event": "...",
  "object": "...",
  "motivation": "..."
}}

Use "Unknown" for any question the evidence cannot answer."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    result = llm_provider.chat_completion(messages, response_model=ContextualItems)

    # If vLLM returns raw string (non-structured path)
    if isinstance(result, str):
        import json, re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                result = ContextualItems(**data)
            except Exception:
                result = ContextualItems()
        else:
            result = ContextualItems()

    print(f"[Contextual] Extracted: {result.model_dump()}")
    return result


# ──────────────────────────────────────────────────────────────
# STEP 2: Map ContextualItems → SVOList (EGMMG format)
# ──────────────────────────────────────────────────────────────

def _map_to_svo(items: ContextualItems) -> SVOList:
    """
    Map the 6 paper context items to EGMMG S-V-O triplets.

    Mapping table:
      People     → Subject node (used as Subject in all other triplets)
      Event      → [People] PERFORMS [Event]
      Location   → [People] LOCATED_IN [Location]
      Date       → [People/Event] OCCURRED_ON [Date]
      Motivation → [People] TARGETS [Motivation]
      Object     → [People/Scene] HAS_STATE [Object]

    Unknown fields are skipped (no triplet generated).
    This prevents hallucinated triplets from polluting the graph.
    """
    subject = items.people if items.people != "Unknown" else "Subject"
    triplets: list[SVOTriplet] = []

    def add(rel: str, obj: str) -> None:
        if obj and obj.strip().lower() != "unknown":
            triplets.append(SVOTriplet(subject=subject, relation=rel, object=obj.strip()))

    add("PERFORMS",    items.event)
    add("LOCATED_IN",  items.location)
    add("OCCURRED_ON", items.date)
    add("TARGETS",     items.motivation)
    add("HAS_STATE",   items.object)

    # Edge case: if people itself is Unknown, add a HAS_STATE for it
    # so RetrievalAgent can still detect "Unverified people claim"
    if items.people == "Unknown":
        triplets.append(SVOTriplet(
            subject="Image",
            relation="HAS_STATE",
            object="people identity unverified",
        ))

    print(f"[Contextual] Mapped to {len(triplets)} S-V-O triplets.")
    return SVOList(triplets=triplets)


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def extract_contextual_svo(
    evidence_list: list[dict],
    visual_entities: list[str],
    source_label: str = "evidence",
) -> tuple[SVOList, ContextualItems]:
    """
    Full Stage 1b pipeline: evidence → 6 context items → S-V-O triplets.

    Args:
        evidence_list:    Top-k ranked evidence from retrieve_evidence().
        visual_entities:  Entity strings from Google Lens.
        source_label:     Label for logging ("evidence" or "caption").

    Returns:
        (svo_list, contextual_items)

        svo_list:          SVOList for RetrievalAgent comparison.
        contextual_items:  Raw ContextualItems for logging / ablation study.
    """
    if not evidence_list and not visual_entities:
        print(f"[Contextual] No evidence available for {source_label}. Returning empty SVO.")
        return SVOList(triplets=[]), ContextualItems()

    evidence_context = _build_evidence_context(evidence_list, visual_entities)
    items = _extract_contextual_items(evidence_context, source_label)
    svo = _map_to_svo(items)

    return svo, items


def extract_caption_svo(caption: str) -> tuple[SVOList, ContextualItems]:
    """
    Extract S-V-O from caption using the SAME 6-question QA format.

    This ensures claim graph and evidence graph use identical schema,
    making RetrievalAgent comparison valid (apples-to-apples).

    The caption IS the "evidence" for the claim graph — no web retrieval needed.
    """
    print("[Contextual] Extracting context items from caption...")

    # Use caption directly as a single evidence item
    caption_as_evidence = [{"title": "Caption under investigation", "text": caption, "image_captions": [], "clip_score": 1.0}]
    return extract_contextual_svo(caption_as_evidence, [], source_label="caption")