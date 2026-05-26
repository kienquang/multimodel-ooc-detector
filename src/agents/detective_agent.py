# detective_agent.py — Vision Wrapper (Tối ưu cho Pipeline V5)

from typing import Dict
from src.agents.base_agent import BaseAgent
from src.llm_provider import llm_provider

class DetectiveAgent(BaseAgent):

    def run(
        self,
        image_url: str,
        prompt:    str,
    ) -> Dict:
        print("[Detective] Gemma 4 E4B visual investigation...")

        report = llm_provider.vision_completion(image_url, prompt)
        
        print("[Detective] Visual report complete.")
        print("\n" + "═"*70)
        print("🕵️ BÁO CÁO ĐIỀU TRA TỪ DETECTIVE (GEMMA 4)")
        print("═"*70)
        print(report)
        print("═"*70 + "\n")

        return {"deep_analysis": report}