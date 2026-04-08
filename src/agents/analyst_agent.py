"""
analyst_agent.py — Final Verdict Agent

Changes from original:
  1. Removed dead vLLM fallback: llm_provider._parse_structured() already handles
     all JSON parsing. isinstance(result, str) was never True.
  2. Fixed biased default: original defaulted to verdict="Real" on parse failure.
     Now defaults to verdict="Unknown" with confidence=0.0 — forces human review
     instead of silently passing misinformation as Real.
  3. Added verdict post-validation: reject responses where LLM ignores the schema
     (e.g. returns "true"/"false" instead of "Real"/"Fake_OOC").
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict

from src.agents.base_agent import BaseAgent


class AnalystOutput(BaseModel):
    verdict:     str   = Field(..., description="Must be exactly 'Real' or 'Fake_OOC'")
    explanation: str   = Field(..., description="Detailed natural language explanation in English")
    confidence:  float = Field(..., ge=0.0, le=1.0, description="Realistic confidence based on evidence strength")

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: str) -> str:
        """
        Normalize LLM verdict to exactly 'Real' or 'Fake_OOC'.
        LLMs sometimes return 'TRUE', 'false', 'Out-of-Context', etc.
        """
        if not isinstance(v, str):
            return "Unknown"
        mapping = {
            "real":           "Real",
            "true":           "Real",
            "legitimate":     "Real",
            "authentic":      "Real",
            "fake_ooc":       "Fake_OOC",
            "fake ooc":       "Fake_OOC",
            "fake":           "Fake_OOC",
            "false":          "Fake_OOC",
            "out-of-context": "Fake_OOC",
            "out_of_context": "Fake_OOC",
            "ooc":            "Fake_OOC",
            "misinformation": "Fake_OOC",
        }
        normalized = mapping.get(v.strip().lower())
        if normalized is None:
            print(f"[Analyst] WARNING: Unrecognized verdict '{v}'. Returning 'Unknown'.")
            return "Unknown"
        return normalized


class AnalystAgent(BaseAgent):

    def run(self, deep_analysis: str) -> Dict:
        print("[Analyst] Generating final verdict (Structured + Few-shot + CoT)...")

        system_prompt = """You are an objective multimodal misinformation judge.
Your sole input is a cross-modal investigation report from a Detective agent.
You must produce a final verdict based ONLY on the evidence in that report.

Rules:
1. If visual evidence + retrieved articles clearly contradict the caption
   → verdict = "Fake_OOC", confidence >= 0.85
2. If caption is consistent with both visual and retrieved evidence
   → verdict = "Real", confidence >= 0.80
3. If evidence is ambiguous or insufficient
   → use whichever verdict is better supported, confidence 0.50 - 0.74
4. Never guess. Base confidence only on contradiction strength.

Few-shot examples:

EXAMPLE 1 — Fake_OOC
Detective report summary:
  - Caption claims: "Donald Trump speaking at Elysee Palace, Paris, 2025"
  - Visual shows: White House interior, American flags, presidential podium
  - S-V-O conflict: Trump LOCATED_IN "Elysee Palace" vs Evidence: Trump LOCATED_IN "White House"
  - Entity check: "Elysee Palace" NOT CONFIRMED visually
→ {"verdict": "Fake_OOC", "confidence": 0.93, "explanation": "The image shows the White House..."}

EXAMPLE 2 — Real
Detective report summary:
  - Caption claims: "Flooded streets in New York City after Hurricane Sandy, 2012"
  - Visual shows: flooded urban streets, Hurricane Sandy emergency signs, NYC architecture
  - S-V-O conflict: none detected
  - Entity check: "New York City" CONFIRMED, "Hurricane Sandy" CONFIRMED via event type
→ {"verdict": "Real", "confidence": 0.88, "explanation": "Visual evidence aligns with caption..."}

EXAMPLE 3 — Low confidence (ambiguous)
Detective report summary:
  - Caption claims: "Angela Merkel at NATO summit 2019"
  - Visual shows: a woman at a formal conference, flags, but face unclear
  - Entity check: "Angela Merkel" UNCERTAIN (face not confirmed by CLIP)
  - S-V-O: OCCURRED_ON "2019" unverifiable from image
→ {"verdict": "Real", "confidence": 0.61, "explanation": "Cannot confirm identity..."}"""

        user_prompt = f"""Detective investigation report:
---
{deep_analysis}
---

Reason step-by-step:
1. Which S-V-O conflicts are the strongest contradictions?
2. Does the visual evidence confirm or deny the entities in the caption?
3. Is the overall contradiction strong enough to conclude Fake_OOC?

Then return ONLY a JSON object (no markdown, no extra text):
{{"verdict": "Real" or "Fake_OOC", "explanation": "...", "confidence": 0.0-1.0}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        result = self.llm.chat_completion(messages, response_model=AnalystOutput)

        # llm_provider guarantees a Pydantic object or model_construct() fallback.
        # isinstance(result, str) will never be True — that dead code is removed.
        # If model_construct() was used (parse failed), verdict will be missing field.
        if not hasattr(result, "verdict") or result.verdict is None:
            print("[Analyst] WARNING: Structured parse failed. Flagging for human review.")
            return {
                "verdict":     "Unknown",    # neutral — not biased toward Real or Fake
                "explanation": "Parsing failed. Manual review required.",
                "confidence":  0.0,
            }

        return {
            "verdict":     result.verdict,
            "explanation": result.explanation,
            "confidence":  float(result.confidence),
        }