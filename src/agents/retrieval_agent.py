"""
retrieval_agent.py — Context Difference Reporter (Upgraded JSON + Fallback)

Vai trò: Chỉ đơn thuần so sánh 2 bản báo cáo (Caption vs Evidence) và liệt kê MỌI điểm khác biệt.
KHÔNG cần suy nghĩ sâu. Việc phân định mâu thuẫn thật/giả sẽ được giao cho DetectiveAgent (Gemma 4).
"""

import json
import re
from typing import Dict, List
from src.agents.base_agent import BaseAgent

class RetrievalAgent(BaseAgent):
    def _detect_differences(self, caption_context: dict, evidence_context: dict) -> List[str]:
        caption_text = json.dumps(caption_context, indent=2, ensure_ascii=False)
        evidence_text = json.dumps(evidence_context, indent=2, ensure_ascii=False)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Compare the 'Caption Context' and 'Evidence Context' across 6 dimensions.\n"
                    "Your job is to report meaningful differences between the two contexts.\n\n"
                    "RULES:\n"
                    "1. IGNORE MISSING INFO: If one side says 'Unknown', 'None', or 'Not mentioned', DO NOT report it as a difference.\n"
                    "2. IGNORE SYNONYMS: If they mean the same thing (e.g., 'Feb 2020' vs '2020'), DO NOT report it.\n"
                    "3. REPORT ACTUAL DIFFERENCES: Focus on details that conflict (e.g., '2022' vs '2020', 'Family' vs 'Police').\n\n"
                    "CRITICAL FORMATTING:\n"
                    "Return a JSON list of strings. You MUST use SINGLE QUOTES (') inside the strings to avoid breaking the JSON. Example:\n"
                    "[\"[DIFFERENCE] Date: Caption claims 'February 2020' but Evidence states '2022'\"]\n\n"
                    "If there are no differences, return exactly []."
                )
            },
            {
                "role": "user",
                "content": f"--- CAPTION CONTEXT ---\n{caption_text}\n\n--- EVIDENCE CONTEXT ---\n{evidence_text}"
            }
        ]

        resp = self.llm.chat_completion(messages)
        raw = resp.choices[0].message.content.strip()

        # Dọn dẹp markdown rác (nếu LLM bọc trong ```json ... ```)
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            # Lớp bảo vệ 1: Parse JSON chuẩn
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                differences = json.loads(match.group(0))
                if isinstance(differences, list):
                    return differences
        except Exception as e:
            # Lớp bảo vệ 2: Cứu hộ (Fallback) khi JSON sập vì dấu ngoặc kép (Unterminated string)
            print(f"⚠️ [Retrieval] JSON parse error: {e}. Kích hoạt Fallback Parser...")
            fallback_diffs = []
            for line in raw.split('\n'):
                # Tìm các dòng có chứa cờ báo hiệu sự khác biệt
                if "[DIFFERENCE]" in line.upper():
                    # Xóa bỏ các ký tự cú pháp JSON (, " ]) ở hai đầu
                    clean_line = re.sub(r'^.*?(\[DIFFERENCE\])', r'\1', line, flags=re.IGNORECASE)
                    clean_line = clean_line.strip('", \']')
                    fallback_diffs.append(clean_line)
            
            if fallback_diffs:
                return fallback_diffs

        return []

    def run(self, caption_context: dict, evidence_context: dict) -> Dict:
        print("🔎 [Retrieval] Spotting differences for Detective Gemma 4...")

        differences = self._detect_differences(caption_context, evidence_context)

        if differences:
            print(f"⚠️  [Retrieval] Found {len(differences)} difference(s) to investigate:")
            for d in differences:
                print(f"   {d}")
        else:
            print("✅ [Retrieval] Contexts match perfectly or differences were trivial.")

        return {"flagged_inconsistencies": differences}