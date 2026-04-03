from serpapi import GoogleSearch
import newspaper
from typing import List, Dict
from src.config import Config

if Config.IS_KAGGLE:
    from sentence_transformers import SentenceTransformer
    _encoder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_evidence(image_url: str, top_k: int = 5) -> List[Dict]:
    print("🔍 [Evidence] Google Lens retrieval...")
    params = {"engine": "google_lens", "url": image_url, "api_key": Config.SERPAPI_API_KEY, "hl": "en"}
    results = GoogleSearch(params).get_dict()

    articles = []
    for match in results.get("visual_matches", [])[:top_k * 3]:
        link = match.get("link")
        if not link: continue
        try:
            article = newspaper.Article(link, language='en')
            article.download()
            article.parse()
            if len(article.text.strip()) > 300:
                articles.append({
                    "url": link,
                    "title": match.get("title", ""),
                    "text": article.text[:2500],
                    "raw_text_for_embedding": match.get("title", "") + " " + article.text[:800]
                })
        except:
            continue

    # === BATCH CLIP reranking chỉ trên Kaggle ===
    if Config.IS_KAGGLE and len(articles) > 1:
        print(f"📐 [CLIP] Batch reranking {len(articles)} articles...")
        doc_texts = [a["raw_text_for_embedding"] for a in articles]
        query_emb = _encoder.encode("image description + caption")  # bạn có thể truyền caption nếu cần
        doc_embs = _encoder.encode(doc_texts, batch_size=16)
        # simple cosine (có thể cải tiến sau)
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        sims = cosine_similarity([query_emb], doc_embs)[0]
        ranked_idx = np.argsort(sims)[::-1]
        articles = [articles[i] for i in ranked_idx[:top_k]]

    print(f"✅ Retrieved {len(articles[:top_k])} high-quality evidences.")
    return articles[:top_k]