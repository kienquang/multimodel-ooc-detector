"""
dataset_evidence_loader.py — VERITE Dataset Loader (V6.0: Pure & Direct Loading)
"""

import json
import re
from pathlib import Path
import pandas as pd

# ──────────────────────────────────────────────────────────────
# VERITE LOADER
# ──────────────────────────────────────────────────────────────

class VERITELoader:
    def __init__(
        self,
        verite_csv:    str,
        articles_csv:  str,  # Vẫn giữ tham số này để file test không bị lỗi khi truyền vào
        image_dir:     str,
        cache_dir:     str = "/kaggle/working/evidence_cache_backup",
    ):
        # 1. CHỈ ĐỌC DUY NHẤT FILE MAIN CSV. BỎ HOÀN TOÀN VIỆC MERGE BẢNG!
        self._df = pd.read_csv(verite_csv)

        # 2. Đảm bảo có cột ID để chạy vòng lặp an toàn
        if "id" not in self._df.columns:
            self._df["id"] = self._df.index.astype(str)

        self._img_dir = Path(image_dir)
        self._cache   = Path(cache_dir)
        
        print(f"[VERITE] {len(self._df)} samples loaded (Pure rows, NO merging).")
        print(f"[VERITE] Cache directory configured at: {self._cache}")

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        for _, row in self._df.iterrows():
            yield row.to_dict()

    def get_image_path(self, sample: dict) -> str:
        img_path = str(sample.get("image_path", ""))
        for candidate in [
            self._img_dir / img_path,
            self._img_dir / Path(img_path).name,
        ]:
            if candidate.exists():
                return str(candidate)

        true_url = str(sample.get("true_url", "")).strip()
        if true_url.startswith("http"):
            return true_url

        print(f"[VERITE] WARNING: Image not found: {img_path}")
        return ""

    def get_caption(self, sample: dict) -> str:
        return str(sample.get("caption", ""))

    get_claim_caption = get_caption

    def get_label(self, sample: dict) -> str:
        return str(sample.get("label", "unknown"))

    def get_evidence(self, sample: dict) -> tuple[list, list]:
        evidence = []
        
        # SỬA Ở ĐÂY: Lấy thẳng chuỗi gốc từ CSV, bỏ qua việc kiểm tra file tồn tại!
        raw_image_path = str(sample.get("image_path", ""))
        img_id = Path(raw_image_path).stem  # Chắc chắn 100% sẽ ra 'true_0', 'false_0'...
        
        json_path = self._cache / f"{img_id}.json"

        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                articles = data if isinstance(data, list) else data.get("evidence", []) or data.get("visual_matches", [])
                
                for art in articles:
                    if "optimized_evidence" in art:
                        evidence.append({
                            "optimized_evidence": art["optimized_evidence"],
                            "source": art.get("source", "unknown"),
                            "clip_score": art.get("clip_score", 0.0),
                            "url": art.get("url", "")
                        })
                    else:
                        ev_texts = []
                        if art.get("page_title"): ev_texts.append(f"Page Title: {art['page_title']}")
                        if art.get("title"): ev_texts.append(f"Title: {art['title']}")
                        if art.get("text"): ev_texts.append(f"Text: {art['text']}")
                            
                        content = "\n".join(ev_texts).strip()
                        if len(content) > 20: 
                            evidence.append({
                                "optimized_evidence": content,
                                "source": art.get("source", "unknown"),
                                "clip_score": art.get("clip_score", 0.0)
                            })
            except Exception as e:
                print(f"⚠️ Lỗi đọc file cache {img_id}.json: {e}")

        visual_entities = _extract_entities(self.get_caption(sample))
        return evidence, visual_entities

# ──────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────
def _extract_entities(text: str) -> list[str]:
    if not text:
        return []
    proper = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    seen, result = set(), []
    for e in proper:
        if e.lower() not in seen and len(e) > 2:
            seen.add(e.lower())
            result.append(e)
    return result[:6]

# (Phần run_batch_evaluation giữ nguyên nếu bạn có dùng ở dưới)
# ──────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────

def run_batch_evaluation(
    loader,
    pipeline_fn,
    max_samples:   int = 1000,
    results_path:  str = "/kaggle/working/verite_results.csv",
) -> pd.DataFrame:
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
            continue

        print(f"\n[Batch] [{i+1}/{len(samples)}] id={sample_id} | label={label}")
        
        try:
            evidence, entities = loader.get_evidence(sample)
            print(f"  evidence: {len(evidence)} item(s) loaded")

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
            
            expected_pred = "Fake" if label in ["out_of_context", "miscaptioned", "out-of-context"] else "True"
            match = "✅" if pred == expected_pred else "❌"
            
            print(f"  {match} pred={pred} ({conf})")

            results.append({
                "sample_id":   sample_id,
                "true_label":  label,
                "verdict":     pred,
                "confidence":  round(float(conf), 3) if isinstance(conf, (int, float)) else conf,
                "explanation": result.get("explanation", "")[:300],
                "has_evidence": len(evidence) > 0,
            })
            done_ids.add(sample_id)
            pd.DataFrame(results).to_csv(results_path, index=False)

        except Exception as e:
            print(f"[Batch] ERROR on {sample_id}: {e}")

    df = pd.DataFrame(results)
    print(f"\n[Batch] Done: {len(df)} samples.")
    return df