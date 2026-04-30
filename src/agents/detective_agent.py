# detective_agent.py — prompt tối ưu cho Gemma 4 instruction-tuned

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.llm_provider import llm_provider


class DetectiveAgent(BaseAgent):

    def run(
        self,
        conflicts:   List[str],   # hard conflicts từ RetrievalAgent
        image_url:   str,
        caption:     str,
    ) -> Dict:
        print("[Detective] Gemma 4 E4B visual investigation...")

        # Không có conflict → không cần visual check
        if not conflicts:
            return {
                "deep_analysis": (
                    "No conflicts flagged by retrieval. "
                    "Image-caption appears consistent without further visual investigation."
                )
            }

        conflict_text = "\n".join(f"- {c}" for c in conflicts)

        # Gemma 4 instruction-tuned: prompt ngắn gọn, rõ ràng
        # Không cần few-shot — E4B đủ mạnh để hiểu task
        prompt = f"""You are a strict Visual Forensics Investigator. Your job is to find HARD EVIDENCE in the image, not guess what might be happening.

═══════════════════════════════════════════
FULL CAPTION (the claim under investigation):
"{caption}"
═══════════════════════════════════════════

EXTERNAL CONFLICTS TO CROSS-CHECK:
{chr(10).join(f"  [{i+1}] {c}" for i, c in enumerate(conflicts)) if conflicts else "  None. Verify caption against image only."}

═══════════════════════════════════════════
STRICT INVESTIGATION RULES:
- You must READ THE FULL CAPTION as a whole sentence, not as isolated keywords.
- You must ONLY report what is DIRECTLY VISIBLE. Never use words like "possibly", "probably", "plausible", "could be", "may be", or "appears to".
- If something is NOT visible in the image, state "NOT VISIBLE — cannot confirm."
- Do NOT fill gaps with assumptions. If the caption says "delivering food" — you must see food. If you see no food, that is a contradiction.
- Do NOT let the conflicts mislead you. Check each conflict against the FULL CAPTION, not against isolated words.

═══════════════════════════════════════════
STEP 1 — SCENE INVENTORY (what is DIRECTLY VISIBLE):
  • Persons: how many, clothing, roles/uniforms if identifiable
  • Actions: what are they physically doing RIGHT NOW in the image
  • Objects: list every visible object relevant to the caption
  • Setting: indoor/outdoor, location cues, time-of-day cues
  • Text in image: any visible signs, labels, banners

STEP 2 — CAPTION vs. IMAGE (verify each key claim):
  For every concrete claim in the Full Caption, state:
  ✅ CONFIRMED — [what you see that proves it]
  ❌ CONTRADICTED — [what you see that disproves it]
  ⚠️ NOT VISIBLE — [cannot confirm either way]

STEP 3 — CONFLICT CROSS-CHECK:
  For each numbered conflict above:
  → Re-read the FULL CAPTION in context.
  → State whether the conflict is a REAL contradiction with the caption, or a FALSE ALARM (the caption already accounts for it).
  → Cite the specific visual detail that resolves it. No assumptions.

STEP 4 — FINAL VERDICT:
  Choose ONE and justify with ONLY visible evidence:
  [TRUE]          — Image directly confirms the Full Caption
  [FAKE]          — Image directly contradicts the Full Caption
  [OUT-OF-CONTEXT] — Image is real but caption misrepresents the situation

Be a detective, not a defense attorney. Do not look for reasons to acquit — look for evidence.
"""
        report = llm_provider.vision_completion(image_url, prompt)
        print("[Detective] Visual report complete.")

        return {"deep_analysis": report}