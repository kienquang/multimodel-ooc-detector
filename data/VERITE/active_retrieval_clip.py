import os
import json
import time
import requests
from io import BytesIO
from urllib.parse import urlparse
from PIL import Image

import trafilatura
from trafilatura.settings import use_config

# Thư viện AI cho Jina-CLIP
import torch
from transformers import AutoModel

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
TEST_IMAGE_URL = "https://images.squarespace-cdn.com/content/5b539de375f9ee8786c67f2d/1581822920950-8IZZLU6YEUBNM9HBYS0F/DSC_5197.jpg?format=1500w&content-type=image%2Fjpeg"

# Giả lập danh sách link cào được từ Google Lens / Vision API
CANDIDATE_URLS = [
    "https://www.chinafashioncollective.com/sheguanghu-2020fw", # Link ít chữ
    "https://www.theguardian.com/world/2020/feb/07/wuhan-facing-wartime-conditions-as-china-tries-to-contain-coronavirus", # Rác
    "https://www.sheguang-hu.com/en/enmrhu/51.html", # Chứa sự thật
    "https://www.nbcnews.com/news/world/video-appears-show-people-china-forcibly-taken-quarantine-over-coronavirus-n1133096" # Link có bối cảnh gốc
]

MAX_PAGES = 3               # Chỉ cần lấy 3 bài báo chất lượng nhất
SIMILARITY_THRESHOLD = 0.22 # Ngưỡng điểm của Jina-CLIP (Tùy chỉnh: 0.20 - 0.28)

_traf_cfg = use_config()
_traf_cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "8")

# ==========================================
# 2. KHỞI TẠO JINA-CLIP
# ==========================================
def init_jina_clip():
    print("🧠 Đang tải mô hình Jina-CLIP (Lần đầu sẽ tải ~1GB từ HuggingFace)...")
    # Tự động chọn GPU nếu có, không thì chạy CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ép buộc tắt bộ nhớ ảo và không cho thư viện tự động chia phần cứng
    model = AutoModel.from_pretrained(
        'jinaai/jina-clip-v1', 
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        device_map=None # CHỐT CHẶN CUỐI CÙNG CHỐNG LỖI META TENSORS
    ).to(device)
    
    model.eval() 
    print(f"✅ Đã tải xong Jina-CLIP trên {device.upper()}!")
    return model, device

# ==========================================
# 3. HÀM CÀO VĂN BẢN (TRAFILATURA)
# ==========================================
def crawl_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            text = trafilatura.extract(resp.text, include_comments=False)
            return text.strip() if text else ""
        return ""
    except Exception:
        return ""

# ==========================================
# 4. LÕI ACTIVE RETRIEVAL (CÀO + CHẤM ĐIỂM)
# ==========================================
def run_active_retrieval(image_url: str, candidate_urls: list, model, device):
    print(f"\n🚀 BẮT ĐẦU QUY TRÌNH ACTIVE RETRIEVAL")
    
    # --- BƯỚC 1: Tải và Mã hóa Hình ảnh (Chỉ làm 1 lần) ---
    print("📸 Đang tải và mã hóa hình ảnh gốc...")
    try:
        img_resp = requests.get(image_url, timeout=10)
        img_resp.raise_for_status()
        image_pil = Image.open(BytesIO(img_resp.content)).convert("RGB")
        
        # Biến bức ảnh thành 1 Vector toán học
        with torch.no_grad():
            img_emb = model.encode_image(image_pil)
            # CHỐT CHẶN: Nếu Jina trả về NumPy Array, ép nó sang PyTorch Tensor
            if not torch.is_tensor(img_emb):
                img_emb = torch.from_numpy(img_emb)
            image_embeds = img_emb.to(device)
            
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")
        return []

    # --- BƯỚC 2: Vòng lặp Cào và Chấm điểm ---
    valid_sources = []
    
    for idx, url in enumerate(candidate_urls):
        if len(valid_sources) >= MAX_PAGES:
            print(f"\n🎯 Đã thu thập đủ {MAX_PAGES} bài báo chất lượng. Dừng cào!")
            break
            
        print(f"\n[{idx+1}/{len(candidate_urls)}] Đang cào: {urlparse(url).netloc}...")
        
        # 1. Cào nội dung
        text = crawl_text(url)
        if not text:
            print("   🔴 Bỏ qua (Trang web chặn hoặc không có chữ).")
            continue
            
        snippet = text[:2500]
        
        # 2. Mã hóa đoạn văn bản thành Vector
        with torch.no_grad():
            txt_emb = model.encode_text([snippet])
            # CHỐT CHẶN: Ép sang PyTorch Tensor cho đồng bộ với ảnh
            if not torch.is_tensor(txt_emb):
                txt_emb = torch.from_numpy(txt_emb)
            text_embeds = txt_emb.to(device)
            
            # 3. Tính độ tương đồng (Cosine Similarity) giữa Ảnh và Text
            sim_score = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
        
        # 4. So sánh với Ngưỡng (Threshold)
        if sim_score >= SIMILARITY_THRESHOLD:
            print(f"   🟢 CHẤP NHẬN (Điểm CLIP: {sim_score:.4f} >= {SIMILARITY_THRESHOLD})")
            valid_sources.append({
                "source": urlparse(url).netloc,
                "url": url,
                "clip_score": round(sim_score, 4),
                "content_snippet": snippet
            })
        else:
            print(f"   🟡 TỪ CHỐI (Lạc đề - Điểm CLIP: {sim_score:.4f} < {SIMILARITY_THRESHOLD})")

    return valid_sources

# ==========================================
# THỰC THI CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Khởi tạo AI
    jina_model, run_device = init_jina_clip()
    
    # Chạy quy trình
    results = run_active_retrieval(TEST_IMAGE_URL, CANDIDATE_URLS, jina_model, run_device)
    
    # In kết quả cuối cùng
    print("\n" + "="*50)
    print("🏆 TỔNG HỢP EVIDENCE ĐÃ LỌC BỞI AI")
    print("="*50)
    print(json.dumps(results, ensure_ascii=False, indent=4))