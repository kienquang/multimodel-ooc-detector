import json
import time
from openai import OpenAI

client = OpenAI(
    api_key="", 
    base_url="https://api.groq.com/openai/v1" 
)

# PROMPT BẤT BẠI: ÉP XUẤT JSON ĐỂ TRÁNH NÓI NGƯỢC VÀ THÊM CHỮ
JSON_PROMPT = """You are a strict data extraction bot. 
Read the Caption (which contains fake news) and the Evidence (which contains the truth).
Extract the core conflicting concept from each. Keep them very short (under 8 words).

You MUST output ONLY a valid JSON object with exactly these two keys:
{
    "fake_concept": "[The false event/fact from the Caption]",
    "true_concept": "[The real event/fact from the Evidence]"
}
"""

FILE_NAME = "verite_fixed_targets_4000.json"

print(f"Đang nạp dữ liệu từ {FILE_NAME}...")
with open(FILE_NAME, "r", encoding="utf-8") as f:
    dataset = json.load(f)

fixed_count = 0

for i, item in enumerate(dataset):
    target = item["target_text"]
    
    # ĐIỀU KIỆN LỌC NHỮNG CÂU BỊ SAI FORM:
    # Nếu target chứa từ thừa thãi như "Caption", "Evidence", "claims", hoặc có dấu xuống dòng "\n"
    if target.startswith("CONTRADICTION:") and ("Caption" in target or "Evidence" in target or "\n" in target or "claims" in target):
        
        user_prompt = f"Caption: {item['input_text']}\nEvidence: {item['input_text']}" 
        # (Lưu ý: input_text của bạn vốn đã chứa cả Caption và Evidence rồi)
        
        try:
            # Ép Groq phải trả về JSON
            response = client.chat.completions.create(
                model="allam-2-7b",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JSON_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            # Đọc JSON và tự động ghép chuỗi bằng Python
            result_json = json.loads(response.choices[0].message.content)
            fake_concept = result_json.get("fake_concept", "").strip()
            true_concept = result_json.get("true_concept", "").strip()
            
            # Cấu trúc A vs B hoàn hảo do chính Python ráp lại
            perfect_target = f"CONTRADICTION: {fake_concept} vs. {true_concept}"
            
            item["target_text"] = perfect_target
            fixed_count += 1
            
            print(f"[{i}] Đã gọt dũa lại: {perfect_target}")
            
        except Exception as e:
            print(f"[{i}] Lỗi: {e}. Bỏ qua...")
            time.sleep(5)
            continue
            
        # Auto-save
        with open(FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
            
        time.sleep(2)

print(f"\n✅ Hoàn tất! Đã ép form thành công {fixed_count} mẫu bị lỗi định dạng.")