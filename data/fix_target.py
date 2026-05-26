import json
import time
import os
from openai import OpenAI

# 1. Khởi tạo API (Dùng con llama-3.1-8b-instant vì task này cực dễ, chạy cho lẹ và rẻ)
client = OpenAI(
    api_key="", 
    base_url="https://api.groq.com/openai/v1" 
)

# 2. System Prompt chuyên biệt cho việc Rút gọn (Summarize Contradiction)
REWRITE_PROMPT = """You are an expert Data Cleaner. 
I will give you a Caption, an Evidence text, and a flawed explanation of their contradiction.
Your ONLY job is to extract the two mutually exclusive concepts and output them in a strict "A vs. B" format.

Rule 1: Keep it extremely short (under 10 words if possible).
Rule 2: Format MUST be exactly: "CONTRADICTION: [Concept from Caption] vs. [Concept from Evidence]"
Rule 3: Output ONLY the string, no quotes, no JSON.

Example Input:
Caption: Peaceful protest...
Evidence: Armed police lockdown...
Flawed Reason: The caption claims a gunman fired shots, but evidence strictly proves no incidents occurred.

Example Output:
CONTRADICTION: Peaceful protest vs. Armed police lockdown
"""

INPUT_FILE = "verite_synthetic_groq_40002.json" # Tên file gốc của bạn
OUTPUT_FILE = "verite_fixed_targets_40002.json"

print(f"Đang đọc dữ liệu từ {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Load file đang làm dở (nếu có)
fixed_dataset = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        fixed_dataset = json.load(f)

start_index = len(fixed_dataset)
print(f"Bắt đầu chạy từ mẫu thứ {start_index}...")

for i in range(start_index, len(dataset)):
    item = dataset[i]
    old_target = item["target_text"]
    
    # BỎ QUA các mẫu không có mâu thuẫn (Tiết kiệm Token tối đa)
    if old_target == "NO_CONTRADICTION":
        fixed_dataset.append(item)
        # Bắt buộc phải lưu ngay để đếm index không bị lệch
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(fixed_dataset, f, ensure_ascii=False, indent=4)
        continue
        
    # CHỈ CHẠY API ĐỂ SỬA CÁC MẪU CONTRADICTION
    user_prompt = f"Caption: {item['input_text']}\nFlawed Reason: {old_target}"
    
    try:
        response = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # Để temperature thấp nhất để nó không sáng tạo thêm
        )
        
        new_target = response.choices[0].message.content.strip()
        item["target_text"] = new_target
        
        print(f"[{i}] Đã sửa: {new_target}")
        
        # CHỈ LƯU khi API gọi thành công, tránh lưu đè dữ liệu hỏng
        fixed_dataset.append(item)
        
        # Auto-save sau mỗi bước
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(fixed_dataset, f, ensure_ascii=False, indent=4)
            
        # Delay nhẹ để không đụng trần API
        time.sleep(2)
        
    except Exception as e:
        # CƠ CHẾ DỪNG KHẨN CẤP KHI HẾT TOKEN
        print(f"\n🚨 [DỪNG KHẨN CẤP] Lỗi/Hết Token tại mẫu thứ {i}: {e}")
        print("="*50)
        print("ĐÂY LÀ INPUT_TEXT CỦA MẪU CUỐI CÙNG BỊ DỪNG:")
        print(item['input_text'])
        print("="*50)
        print(f"💡 Dữ liệu đã được bảo toàn an toàn. Lần chạy tới, code sẽ tự động chạy tiếp từ mẫu {i} này.")
        
        # Lệnh break để thoát hẳn khỏi vòng lặp, không chạy thêm mẫu nào nữa
        break 

# Báo cáo khi chạy xong toàn bộ
if len(fixed_dataset) == len(dataset):
    print("\n✅ Đã tái chế xong toàn bộ dữ liệu!")