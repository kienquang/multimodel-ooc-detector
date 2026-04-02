from src.agents.base_agent import BaseAgent
from pydantic import BaseModel, Field
from typing import Dict
import re

class AnalystOutput(BaseModel):
    """Structured output cho Analyst Agent"""
    verdict: str = Field(..., description="Must be exactly 'Real' or 'Fake_OOC'")
    explanation: str = Field(..., description="Detailed natural language explanation in English")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Realistic confidence based on evidence strength")

class AnalystAgent(BaseAgent):
    """Analyst Agent với Structured Output + Few-shot + CoT (triệt để)"""

    def run(self, deep_analysis: str) -> Dict:
        print("⚖️  [Analyst] Final verdict + confidence (Structured + Few-shot)...")

        system_prompt = """You are an objective multimodal misinformation judge.
You MUST follow these rules strictly:

1. If visual evidence + retrieved articles clearly contradict the caption → verdict = "Fake_OOC" and confidence >= 0.85
2. If everything is consistent → verdict = "Real" and confidence >= 0.80
3. Never guess. Base confidence only on how strong the contradiction is.

Few-shot examples:

Example 1 (Fake_OOC):
Visual: White House background, American flags, man at podium.
Caption: "Donald Trump dancing at France in 2025"
→ verdict: "Fake_OOC", confidence: 0.92

Example 2 (Real):
Visual: Flooded New York streets, Hurricane Sandy signs.
Caption: "Flooded streets in New York City after Hurricane Sandy in 2012"
→ verdict: "Real", confidence: 0.88

Now analyze the following case."""

        user_prompt = f"""Deep analysis from Detective:
{deep_analysis}
Think step-by-step: compare the visual elements with the caption and retrieved evidence before deciding the verdict.
Return ONLY clean JSON (no markdown, no extra text)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # === STRUCTURED OUTPUT (đây là kỹ thuật triệt để) ===
        result = self.llm.chat_completion(messages, response_model=AnalystOutput)

        # vLLM fallback
        if isinstance(result, str):
            # Parse thủ công nếu vLLM
            import json
            try:
                data = json.loads(result)
                result = AnalystOutput(**data)
            except:
                result = AnalystOutput(verdict="Real", explanation=result[:300], confidence=0.6)

        return {
            "verdict": result.verdict,
            "explanation": result.explanation,
            "confidence": float(result.confidence)
        }