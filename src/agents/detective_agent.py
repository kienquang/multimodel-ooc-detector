from src.agents.base_agent import BaseAgent
from src.utils import get_image_description
from typing import Dict, List
from src.config import Config

class DetectiveAgent(BaseAgent):
    """EXCLAIM-style: Visual verification"""

    def run(self, inconsistencies: List[str], image_url_or_path: str, caption: str) -> Dict:
        print("🕵️  [Detective] Visual investigation...")
        img_desc = get_image_description(image_url_or_path)
        prompt = f"""Image visuals: {img_desc}
Caption claim: {caption}
Inconsistencies: {inconsistencies}

Analyze which evidence is correct by matching visual cues (clothes, background, flags, location, objects). Output detailed report."""
        resp = self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return {"deep_analysis": resp.choices[0].message.content}