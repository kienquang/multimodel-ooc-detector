from src.agents.base_agent import BaseAgent
from src.schemas import SVOList
from typing import Dict, List, Tuple
from src.config import Config

class RetrievalAgent(BaseAgent):
    """EGMMG-style: Normalize-First + triplet matching"""

    def _normalize_entity(self, entity: str) -> str:
        """Canonical normalization (Trump == Tổng thống Mỹ)"""
        resp = self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    "Normalize to canonical English name only. "
                    "Examples: 'Tổng thống Mỹ'→'Donald Trump', 'Washington'→'Washington DC'.\n"
                    f"Entity: {entity}\nReturn ONLY the name."
                )
            }],
            temperature=0.0,
            max_tokens=30
        )
        return resp.choices[0].message.content.strip()

    def _normalize_svo(self, svo: SVOList) -> List[Tuple[str, str, str]]:
        """Normalize-First: chuẩn hóa từng triplet riêng lẻ"""
        return [
            (
                self._normalize_entity(t.subject),
                t.relation,
                self._normalize_entity(t.object)
            )
            for t in svo.triplets
        ]

    def run(self, claim_svo: SVOList, evidence_svo: SVOList) -> Dict:
        print("🔎 [Retrieval] Normalize-First (EGMMG style)...")
        claim_norm = self._normalize_svo(claim_svo)
        evidence_norm = self._normalize_svo(evidence_svo)

        claim_dict = {(sub, rel): obj for sub, rel, obj in claim_norm}

        inconsistencies = []
        for sub, rel, obj in evidence_norm:
            key = (sub, rel)
            if key in claim_dict and claim_dict[key] != obj:
                inconsistencies.append(
                    f"CONFLICT: {sub} {rel} → Claim:'{claim_dict[key]}' vs Evidence:'{obj}'"
                )

        return {"flagged_inconsistencies": inconsistencies}