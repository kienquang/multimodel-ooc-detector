from src.agents.base_agent import BaseAgent
from typing import Dict
from src.config import Config

class AnalystAgent(BaseAgent):
    """Final judge - EXCLAIM style"""

    def run(self, deep_analysis: str) -> str:
        print("⚖️  [Analyst] Final verdict...")
        prompt = f"""You are the final judge.
{deep_analysis}

Return ONLY valid JSON:
{{
  "verdict": "Real" or "Fake_OOC",
  "explanation": "detailed natural language explanation in English"
}}"""
        resp = self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return resp.choices[0].message.content