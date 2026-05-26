"""
evidence_agent.py — Sequential Two-Request Evidence Agent

Kiến trúc:
    Request 1 (Temporal Agent): Kiểm tra thời gian → nếu CONTRADICTION thì short-circuit.
    Request 2 (Semantic Agent): Kiểm tra sự kiện / địa điểm / actors → INSUFFICIENT | CONTRADICTION | NO_CONTRADICTION.

Số LLM calls:
    - 1 call  → temporal short-circuit
    - 2 calls → semantic kết luận (CONTRADICTION hoặc NO_CONTRADICTION)
    - 2 calls → semantic trả INSUFFICIENT (Tầng 3 sẽ tiếp quản)
"""

import json
import re
from typing import Dict, Optional
from src.agents.base_agent import BaseAgent
from pydantic import BaseModel, Field


# 1. ĐỊNH NGHĨA LƯỚI BẢO VỆ PYDANTIC CHO SEMANTIC
class SemanticVerdict(BaseModel):
    reasoning: str = Field(
        default="", 
        description="Step-by-step logic comparing the CAPTION vs EVIDENCE context. Focus strictly on the provided text."
    )
    verdict: str = Field(
        default="INSUFFICIENT", 
        description="Must be exactly one of: CONTRADICTION, NO_CONTRADICTION, or INSUFFICIENT"
    )
class EvidenceAgent(BaseAgent):

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def _parse_evidence(self, optimized_evidence_str: str) -> Dict[str, Optional[str]]:
        """
        Parse chuỗi optimized_evidence_str có cấu trúc tiêu chuẩn:
            [PUBLISHED DATE]: YYYY-MM-DD
            [TITLE]: ...
            [CONTENT] / [BEST EVIDENCE CHUNK] / [METADATA/INTRO]: ...
        """
        parsed: Dict[str, Optional[str]] = {
            "published_date": None,
            "title":          "",
            "content":        "",
        }

        date_match = re.search(
            r'\[PUBLISHED DATE\]:\s*(\d{4}-\d{2}-\d{2})',
            optimized_evidence_str,
        )
        if date_match:
            parsed["published_date"] = date_match.group(1)

        title_match = re.search(
            r'\[TITLE\]:\s*(.+?)(?=\n\[|$)',
            optimized_evidence_str,
            re.DOTALL,
        )
        if title_match:
            parsed["title"] = title_match.group(1).strip()

        chunk_match = re.search(
            r'\[BEST EVIDENCE CHUNK\]:\s*(.+?)(?=\n\[|$)',
            optimized_evidence_str,
            re.DOTALL,
        )
        if chunk_match:
            parsed["content"] = chunk_match.group(1).strip()
        else:
            meta_match = re.search(
                r'\[(?:CONTENT|METADATA/INTRO)\]:\s*(.+?)(?=\n\[|$)',
                optimized_evidence_str,
                re.DOTALL,
            )
            if meta_match:
                parsed["content"] = meta_match.group(1).strip()

        return parsed

    def _build_evidence_block(self, parsed: Dict[str, Optional[str]]) -> str:
        parts = []
        if parsed["published_date"]:
            parts.append(f"[PUBLISHED DATE]: {parsed['published_date']}")
        if parsed["title"]:
            parts.append(f"[TITLE]: {parsed['title']}")
        if parsed["content"]:
            parts.append(f"[CONTENT]:\n{parsed['content']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Request 1 — Temporal Agent                                           #
    # ------------------------------------------------------------------ #

    def _temporal_check(self, caption: str, evidence_block: str, published_date: Optional[str]) -> Dict:
        """
        Chỉ kiểm tra chiều thời gian.
        Ưu tiên event date được đề cập trong nội dung bài báo.
        Fallback: Temporal Upper Bound — nếu claim date STRICTLY SAU published date → CONTRADICTION.
        """
        pub_note = (
            f"\nArticle PUBLISHED DATE (for Temporal Upper Bound fallback only): {published_date}"
            if published_date
            else "\nArticle PUBLISHED DATE: NOT AVAILABLE"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a TEMPORAL VERIFICATION specialist. Your ONLY task is to check "
                    "whether the date/time claimed in the RAW CAPTION is consistent with the EVIDENCE.\n\n"

                    "STRICT RULES:\n"
                    "1. PRIORITY — EXPLICIT EVENT DATE ISOLATION:\n"
                    "   - Scan the [TITLE], [METADATA/INTRO], and [CONTENT] to identify the ACTUAL DATE/YEAR the event occurred.\n"
                    "   - Fact-checking articles frequently mention both a 'falsely claimed date' and the 'true event date'. You MUST isolate and extract the TRUE event date.\n"
                    "   - Compare this true event date directly with the caption's date. If they represent mutually exclusive timeframes, return CONTRADICTION. If they align, return NO_CONTRADICTION.\n\n"
                    
                    "2. FALLBACK — TEMPORAL UPPER BOUND (Apply ONLY if NO explicit event date is found):\n"
                    "   - If Caption Date is BEFORE or EQUAL TO the [PUBLISHED DATE]: Return NO_CONTRADICTION (Retrospective reporting is logically valid).\n"
                    "   - If Caption Date is STRICTLY AFTER the [PUBLISHED DATE]: Return CONTRADICTION (Information asymmetry: an article cannot report an event that has not yet occurred).\n\n"
                    
                    "3. NO DATE IN CAPTION:\n"
                    "   - If the caption contains no temporal claim, return NO_CONTRADICTION immediately.\n\n"

                    "OUTPUT — exactly one valid JSON object, no text outside braces:\n"
                    "{\n"
                    "  \"caption_date_claim\": \"Date/period asserted in caption, or 'NONE'\",\n"
                    "  \"evidence_event_date\": \"Event date found in [TITLE] or [CONTENT], or 'NOT_FOUND'\",\n"
                    "  \"temporal_upper_bound_used\": true | false,\n"
                    "  \"reasoning\": \"Explain the deductive logic used without assumptions.\",\n"
                    "  \"verdict\": \"CONTRADICTION\" | \"NO_CONTRADICTION\"\n"
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"--- RAW CAPTION ---\n{caption}\n\n"
                    f"--- EVIDENCE ---\n{evidence_block}"
                    f"{pub_note}"
                ),
            },
        ]

        resp = self.llm.chat_completion(messages)
        raw  = resp.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                verdict = data.get("verdict", "NO_CONTRADICTION")

                print(f"   [Temporal] Caption date  : {data.get('caption_date_claim', 'N/A')}")
                print(f"   [Temporal] Evidence date : {data.get('evidence_event_date', 'N/A')}")
                print(f"   [Temporal] UB fallback   : {data.get('temporal_upper_bound_used', False)}")
                print(f"   [Temporal] Reasoning     : {data.get('reasoning', '')[:120]}")
                print(f"   [Temporal] Verdict       : {verdict}")

                return {
                    "verdict":         verdict,
                    "layer":           "temporal",
                    "published_date":  published_date,
                    "contradictions":  (
                        [data.get("reasoning", "Temporal contradiction detected")]
                        if verdict == "CONTRADICTION" else []
                    ),
                    "temporal_detail": data,
                }
        except Exception as e:
            print(f"⚠️  [Temporal] JSON parse error: {e}")

        # Fallback an toàn
        upper_raw = raw.upper()
        verdict   = (
            "CONTRADICTION"
            if "CONTRADICTION" in upper_raw and "NO_CONTRADICTION" not in upper_raw
            else "NO_CONTRADICTION"
        )
        return {
            "verdict":        verdict,
            "layer":          "temporal_fallback",
            "published_date": published_date,
            "contradictions": ["Temporal contradiction (fallback parser)"] if verdict == "CONTRADICTION" else [],
        }

    # ------------------------------------------------------------------ #
    # Request 2 — Semantic Agent                                           #
    # ------------------------------------------------------------------ #

    def _semantic_check(self, caption: str, evidence_block: str) -> Dict:
        # 2. KHAI BÁO BỘ PROMPT HOÀNG KIM CỦA BẠN
        SYSTEM_PROMPT = (
            "You are a semantic consistency classifier.\n\n"

            "TASK\n"
            "Given a CAPTION and one or more EVIDENCE passages, "
            "determine whether the caption's core factual claim "
            "is consistent with the facts established in the evidence.\n\n"

            "VERDICT OPTIONS\n"
            "NO_CONTRADICTION\n"
            "  The evidence explicitly addresses the specific event or image "
            "in the caption, and its description is compatible with "
            "the caption's core claim.\n\n"
            "CONTRADICTION\n"
            "  The evidence explicitly addresses the specific event or image "
            "in the caption, and states a fact that is logically incompatible "
            "with the caption's core claim. Both statements cannot "
            "simultaneously be true about the same subject.\n\n"
            "INSUFFICIENT\n"
            "  Use this verdict in either of these cases:\n"
            "  (i)  The evidence pertains to a different event, entity, "
            "or domain and cannot verify or refute the caption.\n"
            "  (ii) The evidence discusses the same general subject but "
            "provides only background or biographical information without "
            "describing the specific depicted event or image. "
            "General background about a subject does not constitute "
            "evidence about a specific depicted instance.\n\n"

            "RESOLUTION GUIDELINES\n"
            "Apply these before concluding CONTRADICTION:\n"
            "  (a) Two terms are synonymous only if they are functionally "
            "interchangeable in this specific context — verify they refer "
            "to the same underlying concept, not merely a related one.\n"
            "  (b) A specific term is compatible with its broader category "
            "only if it is a verified instance of that category.\n"
            "  (c) A sub-location is compatible with its containing location.\n"
            "  (d) Silence in the evidence about a peripheral caption detail "
            "is not a contradiction.\n"
            "  (e) CONTRADICTION requires affirmative proof of incompatibility, "
            "not merely absence of confirmation.\n\n"

            "EPISTEMIC PRIOR\n"
            "All three verdicts are equally valid outcomes. "
            "Do not treat any verdict as a default."
        )

        USER_TEMPLATE = (
            "CAPTION: {caption}\n\n"
            "EVIDENCE:\n{evidence}\n\n"
            "Evaluate step by step:\n"
            "Step 1 — State the caption's single core claim.\n"
            "Step 2 — Does the evidence specifically describe the event "
            "or image the caption refers to, or does it only provide "
            "general background about the same subject? "
            "If general background only, verdict is INSUFFICIENT.\n"
            "Step 3 — State what the evidence says about the specific "
            "depicted event or image.\n"
            "Step 4 — Is there a fact in Step 3 that makes the core claim "
            "in Step 1 impossible? If yes, identify it precisely. "
            "If no, state why the claims are compatible.\n"
            "Step 5 — Apply resolution guidelines (a)–(e) to any apparent "
            "conflict. For guideline (a), explicitly verify whether the two "
            "terms are functionally interchangeable or merely related."
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_TEMPLATE.format(caption=caption, evidence=evidence_block)
            }
        ]

        # 3. GỌI LLM CÙNG VỚI RESPONSE_MODEL (ĐỂ KHÓA HỌNG ẢO GIÁC JSON)
        try:
            resp = self.llm.chat_completion(messages, response_model=SemanticVerdict)
            
            # Xử lý an toàn nếu model fallback về chuỗi thô
            if hasattr(resp, 'choices'):
                raw = resp.choices[0].message.content.upper()
                verdict = "CONTRADICTION" if "CONTRADICTION" in raw else "NO_CONTRADICTION" if "NO_CONTRADICTION" in raw else "INSUFFICIENT"
                reasoning = raw[:1000]
            else:
                verdict = resp.verdict.strip().upper()
                reasoning = resp.reasoning.strip()

            # Fix các trường hợp in ra thừa/thiếu
            if verdict not in ["CONTRADICTION", "NO_CONTRADICTION", "INSUFFICIENT"]:
                verdict = "INSUFFICIENT"

            print(f"   [Semantic] Reasoning   : {reasoning[:800]}...")
            print(f"   [Semantic] Verdict     : {verdict}")

            return {
                "verdict":         verdict,
                "layer":           "semantic",
                "contradictions":  [reasoning] if verdict == "CONTRADICTION" else [],
            }

        except Exception as e:
            print(f"⚠️  [Semantic] Agent crashed: {e}")
            return {"verdict": "INSUFFICIENT", "layer": "semantic_fallback", "contradictions": []}
    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        caption:                str,
        optimized_evidence_str: str,
        agent_name:             str = "EvidenceAgent",
        **kwargs,
    ) -> Dict:
        """
        Chạy 2 request tuần tự trên một evidence.
        Returns verdict in {CONTRADICTION, NO_CONTRADICTION, INSUFFICIENT}
        với metadata về path đã đi.
        """
        print(f"\n─── [{agent_name}] Bắt đầu phân tích ─────────────────────")

        parsed         = self._parse_evidence(optimized_evidence_str)
        evidence_block = self._build_evidence_block(parsed)

        # ── Request 1: Temporal ──────────────────────────────────── #
        print(f"\n   ▶ Request 1 / Temporal Agent")
        temporal = self._temporal_check(
            caption        = caption,
            evidence_block = evidence_block,
            published_date = parsed["published_date"],
        )

        if temporal["verdict"] == "CONTRADICTION":
            print(f"   ⚡ Temporal short-circuit → CONTRADICTION (skipping Semantic)")
            return {
                "verdict":        "CONTRADICTION",
                "path":           "temporal_short_circuit",
                "llm_calls":      1,
                "published_date": parsed["published_date"],
                "contradictions": temporal["contradictions"],
                "temporal":       temporal,
                "semantic":       None,
            }

        # ── Request 2: Semantic ──────────────────────────────────── #
        print(f"\n   ▶ Request 2 / Semantic Agent")
        semantic = self._semantic_check(caption, evidence_block)

        final_verdict = semantic["verdict"]   # CONTRADICTION | NO_CONTRADICTION | INSUFFICIENT

        icons = {
            "CONTRADICTION":    "⚠️ ",
            "NO_CONTRADICTION": "✅ ",
            "INSUFFICIENT":     "❓ ",
        }
        print(f"\n   {icons.get(final_verdict, '❓')} [{agent_name}] Final: {final_verdict}")

        return {
            "verdict":        final_verdict,
            "path":           f"temporal_ok__semantic_{final_verdict.lower()}",
            "llm_calls":      2,
            "published_date": parsed["published_date"],
            "contradictions": semantic["contradictions"],
            "temporal":       temporal,
            "semantic":       semantic,
        }