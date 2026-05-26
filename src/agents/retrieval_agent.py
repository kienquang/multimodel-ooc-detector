"""
retrieval_agent.py — Context Difference Reporter (Chain-of-Thought JSON)

Vai trò: So sánh Raw Caption tự do và Evidence (Toàn bộ Raw Text).
Sử dụng kỹ thuật Chain-of-Thought (CoT) để ép LLM phân tích tách bạch, chống ảo giác (Hallucination) 
do đọc nhầm các trích dẫn tin giả trong bài báo Fact-check và chống bắt bẻ ngữ pháp/từ vựng.
"""

import json
import re
from typing import Dict, List
from src.agents.base_agent import BaseAgent

class RetrievalAgent(BaseAgent):
    def _detect_differences(self, raw_caption: str, evidence_context: dict) -> List[str]:
        
        # Chỉ lấy Raw Text, vứt bỏ toàn bộ logic Hints gây nhiễu
        raw_text = evidence_context.get("raw_text", "No raw text available.")

        messages = [
        {
            "role": "system",
            "content": (
            "You are a fact-verification inference engine. "
            "Perform a rigorous entailment and contradiction analysis "
            "between a target [CLAIM] and retrieved [EVIDENCE].\n\n"
        
            "AXIOMS:\n"
            "1. REFERENCE BASELINE: The [EVIDENCE] represents the best available "
            "factual account of the depicted event. Treat it as the primary "
            "reference baseline, while acknowledging it may be incomplete.\n"
            "2. MISINFORMATION ISOLATION: Fact-checking articles may quote false "
            "rumors before debunking them. Extract only the author's verified "
            "conclusions as factual baseline; discard quoted rumors.\n"
            "3. MUTUALLY EXCLUSIVE CONTRADICTION: Flag a contradiction only if "
            "the core narrative, temporal data, spatial data, or primary entities "
            "in the [CLAIM] and [EVIDENCE] are logically irreconcilable.\n"
            "4. TEMPORAL DIMENSION ISOLATION: Distinguish between event_date "
            "(when the event occurred) and publication_date or share_date "
            "(when content was published or redistributed). A mismatch across "
            "different temporal sub-dimensions is NOT a contradiction.\n"
            "5. SEMANTIC EQUIVALENCE: Do not flag lexical variation. "
            "Phrases describing the same action or entity are equivalent.\n"
            "6. INFORMATION ASYMMETRY: Details present in the [CLAIM] but absent "
            "from the [EVIDENCE] constitute unverified information, not contradiction.\n\n"
        
            "OUTPUT: Return one strictly valid JSON object, no text outside it:\n"
            "{\n"
            "  \"claim_core_extraction\": \"\",\n"
            "  \"evidence_factual_baseline\": \"\",\n"
            "  \"logical_alignment_analysis\": \"\",\n"
            "  \"identified_exclusions\": [\n"
            "    {\"dimension\": \"CORE_EVENT|EVENT_DATE|EVENT_LOCATION|KEY_ACTOR\",\n"
            "     \"claim_value\": \"\", \"evidence_value\": \"\"}\n"
            "  ],\n"
            "  \"verdict\": \"CONTRADICTION|NO_CONTRADICTION\"\n"
            "}"
        )
        },
    {
        "role": "user",
        "content": f"[CLAIM]:\n{caption}\n\n[EVIDENCE]:\n{evidence_text}"
    }
]

        resp = self.llm.chat_completion(messages)
        raw = resp.choices[0].message.content.strip()

        # Dọn dẹp markdown rác
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            # Lớp bảo vệ 1: Parse JSON Object mới (Chain of Thought)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                
                # ---> IN RA QUÁ TRÌNH TƯ DUY ĐỂ DEBUG <---
                print(f"   [CoT] Bước 1 (Hiểu Caption): {data.get('step1_caption_claim', '')}")
                print(f"   [CoT] Bước 2 (Hiểu Sự thật): {data.get('step2_article_truth', '')}")
                print(f"   [CoT] Bước 3 (Biện luận): {data.get('step3_compatibility_analysis', '')}")
                
                differences = data.get("differences", [])
                if isinstance(differences, list):
                    return differences
        except Exception as e:
            print(f"⚠️ [Retrieval] JSON parse error: {e}. Kích hoạt Fallback Parser...")
            
        # Lớp bảo vệ 2: Cứu hộ (Fallback) khi JSON sập
        fallback_diffs = []
        for line in raw.split('\n'):
            if "[MUTUALLY EXCLUSIVE]" in line.upper() or "[DIFFERENCE]" in line.upper():
                clean_line = re.sub(r'^.*?(\[MUTUALLY EXCLUSIVE\]|\[DIFFERENCE\])', r'\1', line, flags=re.IGNORECASE)
                clean_line = clean_line.strip('", \']')
                # Ép bọc tag [MUTUALLY EXCLUSIVE] để Gemma 4 không dám cãi
                if not clean_line.startswith("[MUTUALLY EXCLUSIVE]"):
                    clean_line = "[MUTUALLY EXCLUSIVE] " + clean_line.replace("[DIFFERENCE]", "").strip()
                fallback_diffs.append(clean_line)
        
        return fallback_diffs

    def run(self, raw_caption: str, evidence_context: dict) -> Dict:
        print("🔎 [Retrieval] Cross-examining Raw Caption vs Raw Articles...")

        differences = self._detect_differences(raw_caption, evidence_context)

        # Lọc trùng lặp và giới hạn số lượng (Chống tràn Context cho Gemma 4)
        unique_diffs = []
        seen = set()
        for d in differences:
            # Nếu LLM không chịu nhả tag, ta ép thêm vào để khống chế Gemma 4
            if not d.startswith("[MUTUALLY EXCLUSIVE]"):
                d = f"[MUTUALLY EXCLUSIVE] {d}"
                
            key = d[:40].lower() 
            if key not in seen:
                seen.add(key)
                unique_diffs.append(d)
        
        final_diffs = unique_diffs[:5]

        if final_diffs:
            print(f"⚠️  [Retrieval] Found {len(final_diffs)} critical contradiction(s):")
            for d in final_diffs:
                print(f"   {d}")
        else:
            print("✅ [Retrieval] No mutually exclusive contradictions found.")

        return {"flagged_inconsistencies": final_diffs}