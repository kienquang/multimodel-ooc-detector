"""
evidence_coordinator.py — Single-Evidence Coordinator

Kiến trúc mới (đơn giản hóa):
    - Chỉ chạy EvidenceAgent trên Evidence 1 (bài báo điểm cao nhất).
    - EvidenceAgent tự xử lý 2 request tuần tự (Temporal → Semantic).
    - Coordinator chuẩn hóa output và thêm metadata path/confidence.

Verdict đầu ra:
    CONTRADICTION     → Tầng 4 kết luận FAKE
    NO_CONTRADICTION  → Tầng 4 kết luận TRUE
    INSUFFICIENT      → Tầng 3 Visual Grounding tiếp quản
    NOT_ENOUGH_INFO   → Không có evidence nào → Tầng 3 Internal Mode
"""

from typing import Dict, List, Optional
from src.agents.evidence_agent import EvidenceAgent


class EvidenceCoordinator:
    """
    Điều phối luồng text-verification.

    Sử dụng:
        coordinator = EvidenceCoordinator(llm)
        result      = coordinator.run(caption, evidence_list)
    """

    def __init__(self, llm=None):
        # llm được inject qua BaseAgent singleton nên không cần truyền thủ công
        self.agent = EvidenceAgent()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self, caption: str, evidence_list: List[Dict]) -> Dict:
        print("\n" + "═" * 60)
        print("🔎 [EvidenceCoordinator] Starting text-layer evaluation")
        print("═" * 60)

        # ── Không có evidence → Internal Mode ────────────────────── #
        if not evidence_list:
            print("[Coordinator] No evidence retrieved → NOT_ENOUGH_INFO")
            return self._result(
                verdict    = "NOT_ENOUGH_INFO",
                path       = "no_evidence",
                confidence = "NONE",
                detail     = {},
                llm_calls  = 0,
            )

        # ── Chỉ đọc evidence đầu tiên (điểm Jina / rerank cao nhất) #
        best_evidence     = evidence_list[0]
        optimized_str     = best_evidence.get("optimized_evidence", "")

        if not optimized_str.strip():
            print("[Coordinator] Evidence 1 is empty → NOT_ENOUGH_INFO")
            return self._result(
                verdict    = "NOT_ENOUGH_INFO",
                path       = "empty_evidence",
                confidence = "LOW",
                detail     = {"raw_evidence": best_evidence},
                llm_calls  = 0,
            )

        # ── Chạy EvidenceAgent (2 requests tuần tự) ──────────────── #
        agent_result = self.agent.run(
            caption                = caption,
            optimized_evidence_str = optimized_str,
            agent_name             = "Agent 1",
        )

        verdict    = agent_result.get("verdict", "INSUFFICIENT")
        llm_calls  = agent_result.get("llm_calls", 0)
        path       = agent_result.get("path", "unknown")

        # ── Mapping verdict → confidence ──────────────────────────── #
        confidence_map = {
            "CONTRADICTION":    "HIGH",
            "NO_CONTRADICTION": "HIGH",
            "INSUFFICIENT":     "LOW",
        }
        confidence = confidence_map.get(verdict, "LOW")

        return self._result(
            verdict    = verdict,
            path       = path,
            confidence = confidence,
            detail     = {"agent_1": agent_result},
            llm_calls  = llm_calls,
        )

    # ------------------------------------------------------------------ #
    # Helper                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _result(
        verdict:    str,
        path:       str,
        confidence: str,
        detail:     Dict,
        llm_calls:  int,
    ) -> Dict:
        print(f"\n{'═' * 60}")
        print(
            f"📋 [Coordinator] FINAL → verdict={verdict} | "
            f"path={path} | confidence={confidence} | llm_calls={llm_calls}"
        )
        print(f"{'═' * 60}\n")
        return {
            "verdict":    verdict,
            "path":       path,
            "confidence": confidence,
            "llm_calls":  llm_calls,
            "detail":     detail,
        }