"""
detective_agent.py — EXCLAIM-style Cross-Modal Visual Investigation

Role in the pipeline:
  Receives the inconsistency report from RetrievalAgent (S-V-O conflicts),
  then cross-verifies each conflict against visual evidence from the image.

EXCLAIM technique applied:
  - Structured cross-modal prompt: explicitly maps text inconsistencies
    against each granularity level of the visual description
  - Chain-of-Thought reasoning: the agent is asked to reason step-by-step
    before concluding, which feeds richer context to AnalystAgent

EGMMG technique applied:
  - The visual description already contains ENTITY CROSS-CHECK output
    (from utils.py CLIP analysis) — entity-level verification is passed
    directly into this agent's reasoning context
"""

from typing import Dict, List

from src.agents.base_agent import BaseAgent
from src.utils import get_image_description


class DetectiveAgent(BaseAgent):
    """
    Cross-modal visual investigation agent.

    Input:
      - inconsistencies: list of [CONFLICT] / [UNVERIFIED] strings from RetrievalAgent
      - image_url_or_path: image source for visual analysis
      - caption: the claim being investigated

    Output:
      - deep_analysis: detailed English investigation report
    """

    def run(
        self,
        inconsistencies: List[str],
        image_url_or_path: str,
        caption: str,
    ) -> Dict:
        print("[Detective] Cross-modal visual investigation (EXCLAIM)...")

        # Get multi-granularity visual description
        # Pass caption -> EGMMG entity probes are derived from it in utils.py
        visual_description = get_image_description(
            image_source=image_url_or_path,
            caption=caption,            # required for EGMMG entity-level probes
        )

        # Format inconsistencies for the prompt
        if inconsistencies:
            inconsistency_text = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(inconsistencies))
        else:
            inconsistency_text = "  No explicit conflicts detected at S-V-O level."

        # EXCLAIM-style cross-modal prompt
        # Explicitly asks the agent to match each inconsistency against each
        # granularity level of the visual description
        prompt = f"""You are a forensic investigator specializing in out-of-context (OOC) misinformation.

Your task is to cross-examine text-level inconsistencies against visual evidence.

---
## CAPTION UNDER INVESTIGATION
{caption}

---
## S-V-O INCONSISTENCIES (from RetrievalAgent)
{inconsistency_text}

---
## VISUAL EVIDENCE (multi-granularity analysis)
{visual_description}

---
## INVESTIGATION INSTRUCTIONS

Reason step-by-step through THREE levels of cross-modal verification:

**STEP 1 — Entity Verification**
For each entity mentioned in the caption and flagged in inconsistencies:
Does the visual evidence confirm or contradict their presence?
Reference the ENTITY-LEVEL and ENTITY CROSS-CHECK sections above.

**STEP 2 — Event Verification**
Does the event type visible in the image match what the caption claims is happening?
Reference the EVENT-LEVEL section above.

**STEP 3 — Scene/Context Verification**
Do the scene details (location, time, atmosphere) align with the caption's claims?
Reference the SCENE-LEVEL section above.

**STEP 4 — Synthesis**
Based on the three steps above, which pieces of evidence support the caption?
Which contradict it? Is the discrepancy sufficient to indicate out-of-context misuse?

Write a detailed investigation report. Be specific. Cite which visual detail
contradicts which text claim. This report will be the sole input to the final verdict."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        resp = self.llm.chat_completion(messages)
        content = resp.choices[0].message.content

        return {"deep_analysis": content}