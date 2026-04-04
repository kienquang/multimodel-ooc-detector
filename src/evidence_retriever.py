from serpapi import GoogleSearch
import newspaper
from typing import List, Dict
from src.config import Config

def retrieve_evidence(image_url: str, top_k: int = 5) -> List[Dict]:
    """Phiên bản đơn giản, không dùng sentence-transformers (tránh lỗi numpy)"""
    print("🔍 [Evidence] Google Lens retrieval (simple version)...")
    
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": Config.SERPAPI_API_KEY,
        "hl": "en"
    }
    results = GoogleSearch(params).get_dict()

    articles = []
    for match in results.get("visual_matches", [])[:top_k * 3]:
        link = match.get("link")
        if not link:
            continue
        try:
            article = newspaper.Article(link, language='en')
            article.download()
            article.parse()
            if len(article.text.strip()) > 300:
                articles.append({
                    "url": link,
                    "title": match.get("title", ""),
                    "text": article.text[:2500]
                })
        except:
            continue

    print(f"✅ Retrieved {len(articles[:top_k])} high-quality evidences.")
    return articles[:top_k]