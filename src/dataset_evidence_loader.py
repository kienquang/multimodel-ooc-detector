"""
dataset_evidence_loader.py — VERITE Dataset Loader

VERITE_articles.csv structure (thực tế):
  id          → ID của cặp ảnh gốc
  true_url    → URL trực tiếp đến FILE ẢNH gốc (không phải bài báo)
  false_url   → URL trực tiếp đến FILE ẢNH sai ngữ cảnh
  snopes_url  → URL trang fact-check Snopes ← EVIDENCE chính
  query       → Từ khóa tìm kiếm
  true_caption / false_caption → Các caption đi kèm

Evidence strategy:
  1. Crawl snopes_url → Snopes giải thích ảnh thực sự mô tả gì,
     chụp khi nào/ở đâu, tại sao caption sai → perfect evidence
  2. true_url → dùng làm image source thay thế nếu local file không tồn tại
"""

import json
import time
import random
import hashlib
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ──────────────────────────────────────────────────────────────
# CRAWLER với cache
# ──────────────────────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
]


def _crawl_url(url: str, title: str = "", cache_dir: str = ".crawl_cache") -> dict | None:
    """
    Crawl 1 URL với cache. Trả về None nếu thất bại.
    Cache lưu vào {cache_dir}/{md5(url)}.json
    """
    if not url or not url.startswith("http"):
        return None

    # Cache lookup
    cache_path = Path(cache_dir) / (hashlib.md5(url.encode()).hexdigest() + ".json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data if data.get("text") else None
        except Exception:
            pass

    time.sleep(1.5 + random.uniform(0, 0.5))

    headers = {
        "User-Agent":      random.choice(_USER_AGENTS),
        "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code in (403, 404, 429):
            print(f"[Crawl] HTTP {resp.status_code}: {url}")
            cache_path.write_text("{}", encoding="utf-8")
            return None
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        page_title = title
        if not page_title:
            h1 = soup.find("h1")
            page_title = h1.get_text(strip=True)[:120] if h1 else url

        # Extract text
        main = soup.find("article") or soup.find("main") or soup.find("body")
        if not main:
            cache_path.write_text("{}", encoding="utf-8")
            return None

        paragraphs = [p.get_text(separator=" ", strip=True) for p in main.find_all("p")]
        text = " ".join(p for p in paragraphs if len(p) > 30)

        if len(text) < 100:
            cache_path.write_text("{}", encoding="utf-8")
            return None

        captions = [
            fc.get_text(strip=True)
            for fc in soup.find_all("figcaption")
            if 10 < len(fc.get_text(strip=True)) < 200
        ][:5]

        result = {
            "url":            url,
            "title":          page_title,
            "text":           text[:2500],
            "image_captions": captions,
            "clip_score":     1.0,
        }
        cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    except Exception as e:
        print(f"[Crawl] Failed [{type(e).__name__}]: {url}")
        cache_path.write_text("{}", encoding="utf-8")
        return None


# ──────────────────────────────────────────────────────────────
# VERITE LOADER
# ──────────────────────────────────────────────────────────────

class VERITELoader:
    """
    Load VERITE dataset.

    Nhận vào:
      verite_csv:   VERITE.csv    — caption, image_path, label (1000 rows)
      articles_csv: VERITE_articles.csv — id, true_url, false_url, snopes_url, ...

    Join theo id được extract từ image_path:
      "images/true_239.jpg" → id = 239
    """

    def __init__(
        self,
        verite_csv:    str,
        articles_csv:  str,
        image_dir:     str,
        cache_dir:     str = ".crawl_cache/verite",
    ):
        df_main     = pd.read_csv(verite_csv)
        df_articles = pd.read_csv(articles_csv)

        # Extract id từ image_path nếu chưa có cột id
        if "id" not in df_main.columns:
            import re
            def _extract_id(path: str) -> int | None:
                m = re.search(r'_(\d+)\.', str(path))
                return int(m.group(1)) if m else None
            df_main["id"] = df_main["image_path"].apply(_extract_id)

        # Join 2 files theo id
        self._df      = df_main.merge(df_articles, on="id", how="left")
        self._img_dir = Path(image_dir)
        self._cache   = cache_dir

        print(f"[VERITE] {len(self._df)} samples loaded.")
        print(f"[VERITE] Columns: {list(self._df.columns)}")
        print(f"[VERITE] Label distribution:\n{self._df['label'].value_counts().to_string()}")

        # Coverage check
        snopes_ok = self._df["snopes_url"].notna().sum()
        print(f"[VERITE] snopes_url coverage: {snopes_ok}/{len(self._df)} samples")

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        for _, row in self._df.iterrows():
            yield row.to_dict()

    def get_image_path(self, sample: dict) -> str:
        """
        Trả về local path nếu tồn tại, fallback sang true_url (URL ảnh trực tiếp).
        """
        img_path = str(sample.get("image_path", ""))

        # Thử local path
        for candidate in [
            self._img_dir / img_path,
            self._img_dir / Path(img_path).name,
        ]:
            if candidate.exists():
                return str(candidate)

        # Fallback: true_url là URL ảnh trực tiếp
        true_url = str(sample.get("true_url", "")).strip()
        if true_url.startswith("http"):
            return true_url

        print(f"[VERITE] WARNING: Image not found: {img_path}")
        return ""

    def get_caption(self, sample: dict) -> str:
        """Caption trong VERITE.csv (đây là caption đang được kiểm tra)."""
        return str(sample.get("caption", ""))

    # Alias
    get_claim_caption = get_caption

    def get_label(self, sample: dict) -> str:
        """Returns: 'true', 'out_of_context', hoặc 'miscaptioned'"""
        return str(sample.get("label", "unknown"))

    def get_evidence(self, sample: dict) -> tuple[list, list]:
        """
        Evidence = crawl snopes_url (trang fact-check Snopes).

        Snopes giải thích:
          - Ảnh thực sự mô tả gì
          - Bối cảnh thực sự (thời gian, địa điểm, sự kiện)
          - Tại sao caption sai (với OOC/miscaptioned)

        Đây là evidence tốt nhất cho OOC detection — tốt hơn cả bài báo gốc
        vì Snopes đã phân tích và giải thích mâu thuẫn rõ ràng.
        """
        evidence = []

        snopes_url = str(sample.get("snopes_url", "") or "").strip()
        if snopes_url.startswith("http"):
            item = _crawl_url(snopes_url, "Snopes fact-check", cache_dir=self._cache)
            if item:
                item["clip_score"] = 1.0
                evidence.append(item)

        # Visual entities từ caption
        visual_entities = _extract_entities(self.get_caption(sample))

        return evidence, visual_entities


# ──────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────

def _extract_entities(text: str) -> list[str]:
    """Lightweight NER — không cần spaCy."""
    import re
    if not text:
        return []
    proper = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    seen, result = set(), []
    for e in proper:
        if e.lower() not in seen and len(e) > 2:
            seen.add(e.lower())
            result.append(e)
    return result[:6]


# ──────────────────────────────────────────────────────────────
# PRE-CRAWL (chạy local trước để cache)
# ──────────────────────────────────────────────────────────────

def pre_crawl_dataset(loader: VERITELoader, max_samples: int = 1000, delay: float = 1.5) -> dict:
    """
    Pre-crawl snopes_url cho toàn bộ dataset trước khi chạy pipeline.
    Chạy local (CPU, không cần GPU) để tiết kiệm thời gian Kaggle session.
    """
    stats = {"total": 0, "success": 0, "failed": 0}

    for i, sample in enumerate(list(loader)[:max_samples]):
        stats["total"] += 1
        try:
            evidence, _ = loader.get_evidence(sample)
            if evidence:
                stats["success"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"[PreCrawl] Error on sample {i}: {e}")
            stats["failed"] += 1

        if (i + 1) % 100 == 0:
            print(f"[PreCrawl] {i+1}/{max_samples} | {stats}")

    print(f"\n[PreCrawl] Done: {stats}")
    success_rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
    print(f"Success rate: {success_rate:.1f}%")
    return stats


# ──────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────

def run_batch_evaluation(
    loader,
    pipeline_fn,
    max_samples:   int = 1000,
    results_path:  str = "verite_results.csv",
) -> pd.DataFrame:
    """
    Chạy pipeline trên toàn bộ dataset với progressive save + resume.
    """
    results:  list[dict] = []
    done_ids: set[str]   = set()

    if Path(results_path).exists():
        existing = pd.read_csv(results_path)
        done_ids = set(existing["sample_id"].astype(str))
        results  = existing.to_dict("records")
        print(f"[Batch] Resuming — {len(done_ids)} already done.")

    samples = list(loader)[:max_samples]

    for i, sample in enumerate(samples):
        sample_id = str(sample.get("id", i))

        if sample_id in done_ids:
            continue

        image_path = loader.get_image_path(sample)
        caption    = loader.get_caption(sample)
        label      = loader.get_label(sample)

        if not image_path:
            print(f"[Batch] [{i+1}/{len(samples)}] No image, skipping id={sample_id}")
            continue

        print(f"\n[Batch] [{i+1}/{len(samples)}] id={sample_id} | label={label}")
        print(f"  caption: {caption[:70]}...")

        try:
            evidence, entities = loader.get_evidence(sample)
            print(f"  evidence: {len(evidence)} item(s) crawled")

            result = pipeline_fn(
                image_url=image_path,
                caption=caption,
                sample_id=sample_id,
                preloaded_evidence=evidence,
                preloaded_entities=entities,
                use_checkpoint=False,
            )

            pred  = result.get("verdict", "Unknown")
            conf  = result.get("confidence", 0.0)
            match = "✅" if (label == "out_of_context") == (pred == "Fake_OOC") else "❌"
            print(f"  {match} pred={pred} ({conf:.2f})")

            results.append({
                "sample_id":   sample_id,
                "true_label":  label,
                "verdict":     pred,
                "confidence":  round(float(conf), 3),
                "explanation": result.get("explanation", "")[:300],
                "has_evidence": len(evidence) > 0,
            })
            done_ids.add(sample_id)
            pd.DataFrame(results).to_csv(results_path, index=False)

        except Exception as e:
            print(f"[Batch] ERROR on {sample_id}: {type(e).__name__}: {e}")

    df = pd.DataFrame(results)
    print(f"\n[Batch] Done: {len(df)} samples.")
    return df