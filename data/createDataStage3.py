import json
import time
import os
from datasets import load_dataset
from openai import OpenAI

# 1. Khởi tạo API với GROQ
client = OpenAI(
    api_key="", 
    base_url="https://api.groq.com/openai/v1" 
)

print("Đang tải dữ liệu XSum...")
dataset = load_dataset("xsum", split="train")

# 2. System Prompt
SYSTEM_PROMPT = """You are an expert NLP Data Synthesizer for a Fact-Checking system.
Your task is to take a True News Article and its True Summary, and generate a FAKE SUMMARY that contains a fundamental, MUTUALLY EXCLUSIVE contradiction.

CRITICAL RULES FOR GENERATING THE FAKE SUMMARY:
1. THE OUT-OF-CONTEXT RULE (Most Important): Pretend the summary is a caption for a photo. Change the core event, location, or context entirely. If the article is about Event A (e.g., a peaceful protest), write a fake summary claiming it is Event B (e.g., a violent riot).
2. FUNDAMENTAL CONTRADICTIONS ONLY: The Fake Summary and the Article CANNOT BOTH BE TRUE under any circumstances.
3. NO SEMANTIC NITPICKING: Do not just change "taken away" to "forcibly dragged". The error must be a core narrative conflict, not a vocabulary change.
4. NO MISSING INFO TRICKS: Do not just add a random detail that isn't mentioned in the article. The detail must explicitly contradict a stated fact in the article.

You MUST output ONLY a valid JSON object with EXACTLY this structure:
{
    "fake_summary": "[Your subtly corrupted summary that completely changes the core event or key facts]",
    "contradiction_reason": "CONTRADICTION: [Briefly explain why they are mutually exclusive, e.g., 'Caption claims X, but evidence strictly proves Y']"
}
"""

def generate_synthetic_sample(article, true_summary):
    user_prompt = f"Source Article (First 250 words):\n{article}\n\nTrue Summary:\n{true_summary}"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3-32b", # Model 8B tốc độ cực cao
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7 
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                wait_time = (attempt + 1) * 3 
                print(f"  [Đụng trần API] Đang chờ {wait_time} giây để hệ thống phục hồi...")
                time.sleep(wait_time)
            elif "tokens per day" in error_str:
                print("\n🚨 CẢNH BÁO: ĐÃ HẾT GIỚI HẠN TOKENS TRONG NGÀY CỦA GROQ!")
                print("Hãy dừng code và quay lại chạy tiếp vào ngày mai nhé. Dữ liệu đã được lưu an toàn.")
                return "LIMIT_REACHED"
            else:
                print(f"Lỗi API không xác định: {e}")
                return None
    
    print("  Đã thử 5 lần nhưng vẫn thất bại, bỏ qua mẫu này.")
    return None

# ==========================================
# CẤU HÌNH AUTO-SAVE & RESUME (CHẠY TIẾP)
# ==========================================
SAVE_FILE = "verite_synthetic_groq_40002.json"
NUM_SAMPLES_TARGET = 4000
START_OFFSET_XSUM = 7095  # Bỏ qua 500 mẫu đầu tiên đã làm ở file trước

final_dataset = []
samples_already_done = 0

# Kiểm tra xem có file đang làm dở không để nạp vào chạy tiếp
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "r", encoding="utf-8") as f:
        final_dataset = json.load(f)
    
    # Cứ 1 bài XSum sinh ra 2 mẫu (1 Đúng, 1 Sai). Suy ra số bài đã xử lý = len / 2
    samples_already_done = len(final_dataset) // 2
    START_OFFSET_XSUM += samples_already_done
    print(f"🔄 Tìm thấy dữ liệu lưu dở! Đã xử lý {samples_already_done} bài.")
    print(f"🚀 Sẽ chạy tiếp từ bài XSum thứ {START_OFFSET_XSUM}...")

# Tính toán số lượng bài còn phải làm
remaining_to_do = NUM_SAMPLES_TARGET - samples_already_done

if remaining_to_do <= 0:
    print("🎉 Bạn đã hoàn thành đủ chỉ tiêu 4000 mẫu rồi!")
else:
    print(f"Bắt đầu gọi Groq API để tạo {remaining_to_do} bài còn lại...")
    
    # Lấy dữ liệu từ vị trí START_OFFSET_XSUM trở đi
    for i, item in enumerate(dataset.select(range(START_OFFSET_XSUM, START_OFFSET_XSUM + remaining_to_do))):
        current_article_index = START_OFFSET_XSUM + i
        
        article_words = item["document"].split()
        if len(article_words) < 50: continue
        
        article_short = " ".join(article_words[:250])
        true_summary = item["summary"]
        
        fake_data = generate_synthetic_sample(article_short, true_summary)
        
        # Bắt tín hiệu hết giới hạn ngày để dừng vòng lặp an toàn
        if fake_data == "LIMIT_REACHED":
            break
            
        if fake_data and "fake_summary" in fake_data and "contradiction_reason" in fake_data:
            
            # --- VŨ KHÍ POST-PROCESSING ---
            reason_text = fake_data['contradiction_reason'].strip()
            
            # Kiểm tra xem nó có quên chữ "CONTRADICTION:" không
            if not reason_text.upper().startswith("CONTRADICTION"):
                # Nếu quên, ta ép cứng nó vào đầu câu
                reason_text = f"CONTRADICTION: {reason_text}"
            # ------------------------------

            # Lưu mẫu Đúng (Giữ nguyên)
            final_dataset.append({
                "input_text": f"Caption: {true_summary}\nEvidence:\nText: {article_short}\nList factual contradictions:",
                "target_text": "NO_CONTRADICTION"
            })
            
            # Lưu mẫu Sai (Dùng biến reason_text đã được dọn dẹp)
            final_dataset.append({
                "input_text": f"Caption: {fake_data['fake_summary']}\nEvidence:\nText: {article_short}\nList factual contradictions:",
                "target_text": reason_text # <--- Chỗ này đổi thành reason_text
            })
            
            # AUTO-SAVE: Ghi đè file ngay lập tức
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=4)
                
            print(f"Tiến độ: {samples_already_done + i + 1}/{NUM_SAMPLES_TARGET} (Đang xử lý bài XSum {current_article_index}) - Đã Auto-save!")
        else:
            print(f"Bỏ qua bài {current_article_index} do lỗi dữ liệu.")
        
        # Phanh cứng an toàn (10 request / phút để tránh lỗi quá tải)
        time.sleep(6) 

print(f"\n✅ Quá trình kết thúc! Tổng số dòng hiện tại trong file JSON: {len(final_dataset)}")