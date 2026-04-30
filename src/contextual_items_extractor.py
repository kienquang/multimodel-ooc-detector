"""
contextual_items_extractor.py — Stage 1b: 6-Dimension Contextual Extraction

Thay vì cố gắng ép dữ liệu vào khung S-V-O lỏng lẻo dễ đứt gãy, 
bản này giữ nguyên dữ liệu ở dạng Key-Value (6 câu hỏi định hướng của bài báo).
Điều này giúp LLM ở Node Retrieval sau này đối chiếu 1-1 dễ dàng và chính xác hơn.
"""

from pydantic import BaseModel, Field
from src.llm_provider import llm_provider

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

_CONTEXT_QUESTIONS: dict[str, str] = {
    "people":     "Who is shown in the image? (Notable individuals, groups, authorities)",
    "location":   "Where was the event in this image taken? (City, Country, Specific place)",
    "date":       "When was the event in this image taken? (Year, Month, Time context)",
    "event":      "Which event is depicted in the image? (The incident or occasion)",
    "object":     "Which significant animals, plants, buildings, or objects are shown?",
    "motivation": "Why was the image taken? (Purpose/Motivation)",
}

# ──────────────────────────────────────────────────────────────
# EVIDENCE PREPARATION
# ──────────────────────────────────────────────────────────────

def _build_evidence_context(
    evidence_list: list[dict],
    visual_entities: list[str],
) -> str:
    """Combine ranked evidence into a single context string for the LLM."""
    lines = []

    if visual_entities:
        lines.append("## VISUAL ENTITIES")
        for ent in visual_entities:
            lines.append(f"  - {ent}")
        lines.append("")

    lines.append("## RETRIEVED EVIDENCE")
    for i, ev in enumerate(evidence_list):
        lines.append(f"### Source {i+1} ({ev.get('source_type', 'unknown')})")
        lines.append(f"Title: {ev.get('title', '')}")

        captions = ev.get("image_captions", [])
        if captions:
            lines.append("Image captions found on page:")
            for cap in captions[:3]:
                lines.append(f"  \"{cap}\"")

        lines.append(f"Article text:\n{ev.get('text', '')[:600]}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# STEP 1: QA for 6 context items
# ──────────────────────────────────────────────────────────────

def _extract_contextual_items(
    evidence_context: str,
    source_label: str = "evidence",
) -> ContextualItems:
    print(f"[Contextual] Extracting 6 context items from {source_label}...")

    questions_block = "\n".join(
        f"  {i+1}. {key.upper()}: {q}"
        for i, (key, q) in enumerate(_CONTEXT_QUESTIONS.items())
    )

    system_prompt = (
        "You are an expert Context Extraction AI. Your task is to extract information "
        "from the provided text based on 6 specific dimensions.\n"
        "Rules:\n"
        "- Keep the extracted answers concise but highly informative.\n"
        "- If the text DOES NOT explicitly state the answer, you MUST output 'Unknown'.\n"
        "- Do not guess or hallucinate.\n"
        "Respond ONLY with a JSON object."
    )

    user_prompt = f"""Using the evidence below, answer these 6 questions:

{questions_block}

---
TEXT:
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
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    result = llm_provider.chat_completion(messages, response_model=ContextualItems)

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
# PUBLIC API (SẠCH SẼ, TRẢ VỀ JSON MODEL)
# ──────────────────────────────────────────────────────────────

def extract_contextual_svo(
    evidence_list: list[dict],
    visual_entities: list[str],
    source_label: str = "evidence",
) -> dict:
    """
    (LƯU Ý: Tên hàm giữ nguyên để file workflow cũ không lỗi, nhưng trả về Dict)
    """
    if not evidence_list and not visual_entities:
        print(f"[Contextual] No evidence available for {source_label}.")
        return ContextualItems().model_dump()

    evidence_context = _build_evidence_context(evidence_list, visual_entities)
    items = _extract_contextual_items(evidence_context, source_label)

    # Chuyển thẳng Model sang dạng Dictionary dễ đọc
    return items.model_dump()


def extract_caption_svo(caption: str) -> dict:
    """
    (LƯU Ý: Tên hàm giữ nguyên để file workflow cũ không lỗi, nhưng trả về Dict)
    """
    print("[Contextual] Extracting 6 items from caption...")
    caption_as_evidence = [{"title": "Caption", "text": caption}]
    return extract_contextual_svo(caption_as_evidence, [], source_label="caption")