"""
retrieval_agent.py — EGMMG-style S-V-O Conflict Detection

Thay đổi so với bản cũ:
  - Normalize entity: 1 LLM call batch cho TẤT CẢ entities (không gọi N lần)
  - Phát hiện mâu thuẫn rõ ràng hơn: kèm theo relation type
  - Log đầy đủ để debug
"""

import json
import re
from typing import Dict, List, Tuple

from src.agents.base_agent import BaseAgent
from src.schemas import SVOList


class RetrievalAgent(BaseAgent):
    """
    Đối chiếu S-V-O triplets giữa caption và evidence.
    Phát hiện mâu thuẫn về Subject, Relation, Object.

    EGMMG insight: Normalize entity trước khi so sánh
    → tránh miss match do cách viết tên khác nhau
    (vd: "Tổng thống Mỹ" vs "Joe Biden" vs "President Biden")
    """

    # ──────────────────────────────────────────
    # NORMALIZE (Batch — 1 LLM call cho tất cả)
    # ──────────────────────────────────────────

    def _normalize_entities_batch(self, entities: List[str]) -> Dict[str, str]:
        """
        Gửi TẤT CẢ entities trong 1 LLM call → tiết kiệm quota đáng kể.

        Bản cũ: 10 triplets × 2 = 20 LLM calls chỉ để normalize.
        Bản mới: 1 LLM call duy nhất.

        Returns:
            Dict {original_entity: normalized_entity}
        """
        if not entities:
            return {}

        # Deduplicate để không normalize trùng lặp
        unique_entities = list(dict.fromkeys(entities))
        entity_list = "\n".join(f"- {e}" for e in unique_entities)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an entity normalization assistant. "
                    "Normalize each entity to its canonical English name. "
                    "Rules:\n"
                    "  - Names in other languages → translate to English common name\n"
                    "  - Aliases → most widely recognized name\n"
                    "  - Unknown entity → keep as-is\n"
                    "Return ONLY a JSON object like: "
                    '{"original": "normalized", ...}\n'
                    "No markdown, no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Normalize these entities:\n{entity_list}",
            },
        ]

        resp = self.llm.chat_completion(messages)
        raw = resp.choices[0].message.content.strip()

        # Parse JSON an toàn
        try:
            # Extract JSON block nếu model wrap trong markdown
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                norm_map = json.loads(match.group(0))
                # Chỉ giữ các entry hợp lệ (str → str)
                norm_map = {
                    str(k): str(v)
                    for k, v in norm_map.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
                print(f"✅ [Normalize] {len(norm_map)}/{len(unique_entities)} entities normalized.")
                return norm_map
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️  [Normalize] JSON parse failed: {e}. Using original entities.")

        # Fallback: giữ nguyên tất cả
        return {e: e for e in unique_entities}

    def _apply_norm(
        self,
        svo: SVOList,
        norm_map: Dict[str, str],
    ) -> List[Tuple[str, str, str]]:
        """Áp dụng norm_map lên toàn bộ triplets."""
        return [
            (
                norm_map.get(t.subject, t.subject),
                t.relation,
                norm_map.get(t.object, t.object),
            )
            for t in svo.triplets
        ]

    # ──────────────────────────────────────────
    # CONFLICT DETECTION
    # ──────────────────────────────────────────

    def _detect_conflicts(
        self,
        claim_triplets: List[Tuple[str, str, str]],
        evidence_triplets: List[Tuple[str, str, str]],
    ) -> List[str]:
        """
        So sánh triplets theo key = (Subject, Relation).
        Nếu cùng key mà Object khác → mâu thuẫn.

        Cũng phát hiện:
        - Entity trong claim nhưng không xuất hiện trong evidence (thiếu xác nhận)
        """
        conflicts = []

        # Build lookup: (subject, relation) → object từ evidence
        evidence_dict: Dict[Tuple[str, str], str] = {}
        for sub, rel, obj in evidence_triplets:
            key = (sub.lower(), rel)
            evidence_dict[key] = obj

        # So sánh từng triplet của claim với evidence
        for sub, rel, obj in claim_triplets:
            key = (sub.lower(), rel)

            if key in evidence_dict:
                ev_obj = evidence_dict[key]
                if ev_obj.lower() != obj.lower():
                    conflicts.append(
                        f"[CONFLICT] {sub} {rel}: "
                        f"Claim='{obj}' | Evidence='{ev_obj}'"
                    )
                # else: consistent → không report
            else:
                # Relation tồn tại trong claim nhưng không có trong evidence
                # Đây là "unverified" — ghi nhận nhẹ hơn CONFLICT
                conflicts.append(
                    f"[UNVERIFIED] {sub} {rel} '{obj}' — not found in evidence"
                )

        return conflicts

    # ──────────────────────────────────────────
    # MAIN
    # ──────────────────────────────────────────

    def run(self, claim_svo: SVOList, evidence_svo: SVOList) -> Dict:
        """
        Args:
            claim_svo:    SVO từ caption (cần kiểm tra).
            evidence_svo: SVO từ bài báo gốc (bằng chứng).

        Returns:
            {"flagged_inconsistencies": List[str]}
        """
        print("🔎 [Retrieval] Batch normalize + conflict detection (EGMMG)...")

        # Collect tất cả entities từ cả 2 SVO để normalize trong 1 lần
        all_entities = (
            [t.subject for t in claim_svo.triplets]
            + [t.object for t in claim_svo.triplets]
            + [t.subject for t in evidence_svo.triplets]
            + [t.object for t in evidence_svo.triplets]
        )

        norm_map = self._normalize_entities_batch(all_entities)

        claim_norm = self._apply_norm(claim_svo, norm_map)
        evidence_norm = self._apply_norm(evidence_svo, norm_map)

        print(f"   Claim triplets:    {len(claim_norm)}")
        print(f"   Evidence triplets: {len(evidence_norm)}")

        conflicts = self._detect_conflicts(claim_norm, evidence_norm)

        if conflicts:
            print(f"⚠️  [Retrieval] {len(conflicts)} issue(s) found:")
            for c in conflicts:
                print(f"   {c}")
        else:
            print("✅ [Retrieval] No conflicts detected.")

        return {"flagged_inconsistencies": conflicts}