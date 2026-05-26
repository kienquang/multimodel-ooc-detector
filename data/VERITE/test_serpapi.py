import os
import json
import time
import serpapi
import requests
import trafilatura

# ==========================================
# 1. CẤU HÌNH API KEY VÀ DANH SÁCH ẢNH TEST
# ==========================================
SERPAPI_API_KEY = "" 

# Hãy điền link ảnh của bạn
TEST_IMAGES = [
    "https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/2018/09/bikers.jpg",
]

# ==========================================
# 2. HÀM CHẠY BENCHMARK & CÀO NỘI DUNG
# ==========================================
def run_serpapi_benchmark(image_urls, api_key):
    client = serpapi.Client(api_key=api_key)
    benchmark_results = []

    print(f"🚀 BẮT ĐẦU TEST SERPAPI LENS & CÀO BÁO ({len(image_urls)} ẢNH)...\n" + "="*50)

    for idx, url in enumerate(image_urls):
        print(f"\n📸 Đang quét Ảnh [{idx + 1}/{len(image_urls)}]: {url[:60]}...")
        start_time = time.time()
        
        try:
            # Bước 1: Gọi Google Lens qua SerpAPI (Tìm Bản đồ)
            results = client.search({
                "engine": "google_lens",
                "url": url,
                "hl": "en"
            })
            results_dict = dict(results)
            
            # Thực thể lõi
            kg = results_dict.get("knowledge_graph", [{}])
            main_entity = kg[0].get("title", "N/A") if isinstance(kg, list) and len(kg) > 0 else "N/A"
            
            visual_matches = results_dict.get("visual_matches", [])
            sources = []
            
            print(f"   ✅ SerpAPI tìm thấy {len(visual_matches)} links. Bắt đầu tìm kiếm 5 bài viết hợp lệ...")
            
            # Bước 2: Lặp qua TẤT CẢ các link cho đến khi đủ 5 link thành công
            for match_idx, match in enumerate(visual_matches): 
                
                # NẾU ĐÃ ĐỦ 5 BÀI -> DỪNG VÒNG LẶP (BREAK)
                if len(sources) >= 5:
                    print("   🎯 Đã nhặt đủ 5 bài viết hợp lệ! Dừng cào ảnh này.")
                    break
                    
                link = match.get("link", "")
                title = match.get("title", "")
                source_name = match.get("source", "")
                
                if not link:
                    continue

                try:
                    # 1. Giả lập trình duyệt
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    
                    # 2. Tải trang web
                    response = requests.get(link, headers=headers, timeout=10)
                    
                    # 3. Chỉ xử lý nếu HTTP 200 (Thành công)
                    if response.status_code == 200:
                        extracted_text = trafilatura.extract(response.text)
                        
                        # 4. Chỉ lưu nếu thực sự có chữ (Có nội dung)
                        if extracted_text:
                            scraped_text = extracted_text.strip()[:1000] 
                            print(f"      [{match_idx+1}] 🟢 Đã lấy được (Giỏ có {len(sources)+1}/5): {source_name}")
                            
                            # CHỈ LƯU VÀO MẢNG KHI CÀO THÀNH CÔNG
                            sources.append({
                                "title": title,
                                "link": link,
                                "source": source_name,
                                "content_snippet": scraped_text
                            })
                        else:
                            print(f"      [{match_idx+1}] 🟡 Bỏ qua (Trang không có chữ): {source_name}")
                    else:
                        print(f"      [{match_idx+1}] 🔴 Bỏ qua (Lỗi {response.status_code}): {source_name}")

                except Exception as scrape_err:
                    print(f"      [{match_idx+1}] 🔴 Bỏ qua (Lỗi kết nối/Timeout): {source_name}")

            latency = round(time.time() - start_time, 2)
            
            # Lưu toàn bộ kết quả của ảnh này vào mảng benchmark_results
            benchmark_results.append({
                "image_id": f"image_{idx+1}",
                "image_url": url,
                "latency_sec": latency,
                "main_entity": main_entity,
                "top_5_sources": sources
            })

        except Exception as e:
            print(f"   ❌ Lỗi truy xuất SerpAPI: {e}")
            benchmark_results.append({
                "image_id": f"image_{idx+1}",
                "error": str(e)
            })

    # ==========================================
    # 3. XUẤT KẾT QUẢ RA FILE JSON
    # ==========================================
    output_file = "serpapi_with_content_benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=4)
    
    print("\n" + "="*50)
    print(f"🎉 HOÀN THÀNH! Đã lưu kết quả (kèm nội dung cào) vào: {output_file}")

# Chạy Script
if __name__ == "__main__":
    run_serpapi_benchmark(TEST_IMAGES, SERPAPI_API_KEY)