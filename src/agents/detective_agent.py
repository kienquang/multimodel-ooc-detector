from src.agents.base_agent import BaseAgent
from src.utils import get_image_description
from typing import Dict, List

class DetectiveAgent(BaseAgent):
    """EXCLAIM-style: Visual verification"""

    def run(self, inconsistencies: List[str], image_url_or_path: str, caption: str) -> Dict:
        print("🕵️  [Detective] Visual investigation...")
        img_desc = get_image_description(image_url_or_path)
        prompt = f"""Image visuals: {img_desc}
Caption claim: {caption}
Inconsistencies: {inconsistencies}

Analyze which evidence is correct by matching visual cues (clothes, background, flags, location, objects). Output detailed report."""

        messages = [{"role": "user", "content": prompt}]
        resp = self.llm.chat_completion(messages)          # ← DÙNG ADAPTER
        content = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
        return {"deep_analysis": content}