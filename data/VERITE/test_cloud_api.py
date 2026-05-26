import os, json, time, requests
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

import trafilatura
from trafilatura.settings import use_config

import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account

load_dotenv()

# ==========================================
# 1. CẤU HÌNH LINK ẢNH CẦN TEST Ở ĐÂY
# ==========================================
TEST_IMAGE_URL = "https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/2018/09/bikers.jpg"
TEST_IMAGE_ID  = "test_image_01" # Tên file JSON sẽ lưu

# ==========================================
# 2. CẤU HÌNH HỆ THỐNG
# ==========================================
CACHE_DIR       = Path(".crawl_cache/single_test")
MAX_QUOTA       = 2000          
MAX_PAGES       = 5            
VISION_SCOPE    = ["https://www.googleapis.com/auth/cloud-vision"]
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

_traf_cfg = use_config()
_traf_cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "8")

SA_FILES = [v for v in [
    os.getenv("VISION_SA_1"),
    os.getenv("VISION_SA_2"),
    os.getenv("VISION_SA_3"),
] if v and Path(v).exists()]

assert SA_FILES, "Không tìm thấy file JSON Service Account! Kiểm tra file .env"

# ── Service Account Key Manager ───────────────────────────────────────────────
class VisionKeyManager:
    def __init__(self):
        self.sa_files  = SA_FILES
        self.idx       = 0
        self._credentials = {}   

    def _get_credentials(self, sa_file: str):
        if sa_file not in self._credentials:
            creds = service_account.Credentials.from_service_account_file(
                sa_file, scopes=VISION_SCOPE
            )
            self._credentials[sa_file] = creds
        creds = self._credentials[sa_file]
        if not creds.valid:
            creds.refresh(google.auth.transport.requests.Request())
        return creds

    @property
    def current_sa(self) -> str:
        return self.sa_files[self.idx]

    def get_token(self) -> str:
        return self._get_credentials(self.current_sa).token

# ── Google Vision API call ────────────────────────────────────────────────────
def call_vision_api(image_url: str, key_mgr: VisionKeyManager) -> dict | None:
    token = key_mgr.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    body = {
        "requests": [{
            "image":    {"source": {"imageUri": image_url}},
            "features": [{"type": "WEB_DETECTION", "maxResults": 30}]
        }]
    }
    try:
        resp = requests.post(VISION_ENDPOINT, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        response_obj = result.get("responses", [{}])[0]
        if "error" in response_obj:
            print(f"  [VISION ERR] {response_obj['error']}")
            return None

        return response_obj.get("webDetection", {})

    except Exception as e:
        print(f"  [ERR] Vision: {e}")
        return None

# ── Trafilatura crawler ───────────────────────────────────────────────────────
def crawl_page_trafilatura(url: str) -> dict:
    empty = {"title": "", "description": "", "text": "", "image_captions": []}
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return empty
            
        downloaded = response.text
        if not downloaded: return empty

        metadata = trafilatura.extract_metadata(downloaded)
        desc = metadata.description if (metadata and metadata.description) else ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""

        if not text and desc: text = desc
        if not text.strip(): return empty

        return {
            "title":          (metadata.title if metadata else ""),
            "description":    desc,
            "text":           text, 
        }
    except Exception:
        return empty

# ── Build cache entry cho 1 sample ────────────────────────────────────────────
def process_single_image(image_id: str, image_url: str, key_mgr: VisionKeyManager) -> dict | None:
    
    start_time = time.time()
    print(f"\n🚀 BẮT ĐẦU CÀO VISION API CHO ẢNH: {image_id}")
    print(f"🔗 URL: {image_url[:80]}...")

    # Step 1: Vision API
    wd = call_vision_api(image_url, key_mgr)
    if wd is None:
        return None

    # Lấy Main Entity
    main_entity = "N/A"
    best_guess_labels = [g["label"] for g in wd.get("bestGuessLabels", [])]
    visual_entities = [e.get("description", "") for e in wd.get("webEntities", []) if e.get("score", 0) > 0.3]
    
    if best_guess_labels:
        main_entity = best_guess_labels[0]
    elif visual_entities:
        main_entity = visual_entities[0]
        
    print(f"  ✅ Vision API trích xuất thực thể chính: {main_entity}")

    # Step 2: Blacklist Filter
    BLACKLIST = [
        "youtube.com", "youtu.be", "instagram.com", "pinterest.com", 
        "buymeacoffee.com", "reddit.com", "x.com", "twitter.com",
        "facebook.com", "fb.com", "fb.watch", "tiktok.com"
    ]
    
    all_pages = wd.get("pagesWithMatchingImages", [])
    clean_pages = [p for p in all_pages if p.get("url") and not any(d in p.get("url").lower() for d in BLACKLIST)]

    if not clean_pages:
        print("  [Warn] Vision API chỉ trả về MXH rác, không có link hợp lệ!")
        return None

    print(f"  ✅ Tìm thấy {len(clean_pages)} link sạch. Bắt đầu cào text...")

    # Step 3: Crawl với Trafilatura (Chạy đến khi đủ 5 bài)
    sources = []
    for match_idx, page in enumerate(clean_pages):
        if len(sources) >= MAX_PAGES:
            break

        page_url   = page.get("url", "")
        page_title = page.get("pageTitle", "")

        crawled = crawl_page_trafilatura(page_url)
        text = crawled.get("text", "")

        if text.strip():
            domain = urlparse(page_url).netloc
            sources.append({
                "title": crawled.get("title") or page_title,
                "link": page_url,
                "source": domain,
                "content_snippet": text.strip()[:2500] 
            })
            print(f"      [{match_idx+1}] 🟢 Lấy thành công (Giỏ: {len(sources)}/{MAX_PAGES}): {domain}")
        else:
            print(f"      [{match_idx+1}] 🔴 Bỏ qua (Không có chữ): {page_url[:40]}...")

    if not sources:
        print("  [Warn] Không cào được văn bản hợp lệ nào!")
        return None

    latency = round(time.time() - start_time, 2)

    # Step 4: Đóng gói
    return {
        "image_id":           image_id,
        "image_url":          image_url,
        "latency_sec":        latency,
        "main_entity":        main_entity,
        "top_5_sources":      sources,
        "vision_best_guess":  best_guess_labels,
        "vision_entities":    visual_entities[:5],
        "crawled_at":         datetime.now(timezone.utc).isoformat()
    }

# ── Thực thi ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_manager = VisionKeyManager()
    
    result = process_single_image(TEST_IMAGE_ID, TEST_IMAGE_URL, key_manager)
    
    if result:
        output_file = CACHE_DIR / f"{TEST_IMAGE_ID}.json"
        output_file.write_text(
            json.dumps([result], ensure_ascii=False, indent=4), # Bọc mảng [] để giống format SerpAPI
            encoding="utf-8"
        )
        print("\n" + "="*60)
        print(f"🎉 HOÀN THÀNH! Mất {result['latency_sec']}s")
        print(f"📂 Kết quả được lưu tại: {output_file.resolve()}")