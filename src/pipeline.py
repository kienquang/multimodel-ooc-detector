"""
pipeline.py — LangGraph Multi-Agent Pipeline

Stage 1 (Paper + EGMMG hybrid):
  1a. retrieve_evidence()         → ranked evidence + visual entities (CLIP image re-ranking)
  1b. extract_caption_svo()       → claim graph S-V-O (from caption, 6-QA format)
  1b. extract_contextual_svo()    → evidence graph S-V-O (from web evidence, 6-QA format)

Stage 2 (EXCLAIM Multi-Agent):
  retrieval_node → detective_node → analyst_node → [self-refine loop] → END
"""
"""
pipeline.py — LangGraph Multi-Agent Pipeline

Hai chế độ hoạt động:

  [DEMO / test nhỏ]
    run_pipeline(image_url, caption)
    → Tự gọi retrieve_evidence() → Google Lens → SerpAPI

  [BATCH / dataset lớn]
    run_pipeline(image_url, caption,
                 preloaded_evidence=..., preloaded_entities=...)
    → Dùng evidence từ dataset metadata → KHÔNG gọi SerpAPI
    → Dùng cho NewsCLIPpings và VERITE ở Kaggle
"""

import json
from pathlib import Path
from typing import TypedDict, Any, Optional

from langgraph.graph import StateGraph, END

from src.config import Config
from src.contextual_items_extractor import extract_caption_svo, extract_contextual_svo
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.detective_agent import DetectiveAgent
from src.agents.analyst_agent   import AnalystAgent


# ──────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    image_url:              str
    caption:                str
    image_bytes:            Optional[bytes]
    visual_entities:        list[str]
    evidence_list:          list
    claim_svo:              Any
    evidence_svo:           Any
    claim_context_items:    Any
    evidence_context_items: Any
    inconsistencies:        list[str]
    deep_analysis:          str
    final_result:           dict
    refine_count:           int


# ──────────────────────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────────────────────

def retrieval_node(state: PipelineState) -> dict:
    print("\n─── [Node: Retrieval] ───────────────────────────")

    evidence_list   = state.get("evidence_list", [])
    visual_entities = state.get("visual_entities", [])

    # Chỉ gọi SerpAPI nếu KHÔNG có preloaded evidence
    if not evidence_list:
        from src.evidence_retriever import retrieve_evidence
        evidence_list, visual_entities = retrieve_evidence(
            image_url=state["image_url"],
            image_bytes=state.get("image_bytes"),
            top_k=Config.MAX_EVIDENCE,
        )

    # Extract claim graph từ caption (6-QA format)
    claim_svo, claim_ctx = extract_caption_svo(state["caption"])

    # Extract evidence graph từ web evidence
    if evidence_list or visual_entities:
        evidence_svo, evidence_ctx = extract_contextual_svo(
            evidence_list=evidence_list,
            visual_entities=visual_entities,
            source_label="evidence",
        )
    else:
        from src.schemas import SVOList
        from src.contextual_items_extractor import ContextualItems
        evidence_svo = SVOList(triplets=[])
        evidence_ctx = ContextualItems()
        print("[Retrieval] No evidence available.")

    # S-V-O conflict detection
    if evidence_svo.triplets:
        inconsistencies = RetrievalAgent().run(claim_svo, evidence_svo)["flagged_inconsistencies"]
    else:
        inconsistencies = ["No web evidence available for S-V-O comparison."]

    return {
        "evidence_list":          evidence_list,
        "visual_entities":        visual_entities,
        "claim_svo":              claim_svo,
        "evidence_svo":           evidence_svo,
        "claim_context_items":    claim_ctx,
        "evidence_context_items": evidence_ctx,
        "inconsistencies":        inconsistencies,
        "refine_count":           state.get("refine_count", 0),
    }


def detective_node(state: PipelineState) -> dict:
    print("\n─── [Node: Detective] ───────────────────────────")
    result = DetectiveAgent().run(
        inconsistencies=state["inconsistencies"],
        image_url_or_path=state["image_url"],
        caption=state["caption"],
    )
    return {"deep_analysis": result["deep_analysis"]}


def analyst_node(state: PipelineState) -> dict:
    print("\n─── [Node: Analyst] ─────────────────────────────")
    result = AnalystAgent().run(state["deep_analysis"])
    return {
        "final_result": result,
        "refine_count": state.get("refine_count", 0) + 1,
    }


def should_refine(state: PipelineState) -> str:
    confidence   = state["final_result"].get("confidence", 0.5)
    refine_count = state.get("refine_count", 0)
    if confidence < Config.REFINE_THRESHOLD and refine_count < Config.MAX_REFINE:
        print(f"[Self-Refine] confidence={confidence:.2f}, loop={refine_count}/{Config.MAX_REFINE}")
        return "detective"
    return END


# ──────────────────────────────────────────────────────────────
# GRAPH
# ──────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(PipelineState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("detective", detective_node)
    graph.add_node("analyst",   analyst_node)
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "detective")
    graph.add_edge("detective", "analyst")
    graph.add_conditional_edges("analyst", should_refine, {"detective": "detective", END: END})
    return graph.compile()


_app = _build_graph()


# ──────────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────────

def _save_checkpoint(sample_id: str, state: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"checkpoint_{sample_id}.json"
    try:
        state_copy = {k: v for k, v in state.items() if k != "image_bytes"}
        with open(path, "w") as f:
            json.dump(state_copy, f, default=str, indent=2)
    except Exception as e:
        print(f"[Checkpoint] Save failed: {e}")


def _load_checkpoint(sample_id: str, out_dir: str) -> Optional[dict]:
    path = Path(out_dir) / f"checkpoint_{sample_id}.json"
    if path.exists():
        with open(path) as f:
            print(f"[Checkpoint] Resuming: {path.name}")
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    image_url:           str,
    caption:             str,
    image_bytes:         bytes | None = None,
    sample_id:           str  = "sample",
    use_checkpoint:      bool = True,
    checkpoint_dir:      str  = "/kaggle/working/checkpoints",
    # Batch mode: pass pre-loaded evidence to skip SerpAPI
    preloaded_evidence:  list | None = None,
    preloaded_entities:  list | None = None,
) -> dict:
    """
    Run detection pipeline for one image-caption pair.

    Demo mode  (preloaded_evidence=None):
        → retrieve_evidence() is called → uses SerpAPI
        → suitable for testing 1-10 samples

    Batch mode (preloaded_evidence=[...]):
        → SerpAPI is skipped entirely
        → suitable for NewsCLIPpings (71k) and VERITE (1k)
        → call via dataset_evidence_loader.run_batch_evaluation()
    """
    Config.log_env()
    Config.validate()

    if use_checkpoint and Config.IS_KAGGLE:
        cached = _load_checkpoint(sample_id, checkpoint_dir)
        if cached and "final_result" in cached:
            return cached["final_result"]

    mode = "batch (preloaded)" if preloaded_evidence is not None else "demo (SerpAPI)"
    print(f"\n[Pipeline] START | id={sample_id} | mode={mode}")
    print(f"  Image:   {str(image_url)[:80]}")
    print(f"  Caption: {caption[:100]}")

    initial_state: PipelineState = {
        "image_url":              image_url,
        "caption":                caption,
        "image_bytes":            image_bytes,
        # Pre-populate if provided → retrieval_node skips SerpAPI
        "evidence_list":          preloaded_evidence or [],
        "visual_entities":        preloaded_entities or [],
        "claim_svo":              None,
        "evidence_svo":           None,
        "claim_context_items":    None,
        "evidence_context_items": None,
        "inconsistencies":        [],
        "deep_analysis":          "",
        "final_result":           {},
        "refine_count":           0,
    }

    final_state = _app.invoke(initial_state)

    if Config.IS_KAGGLE:
        _save_checkpoint(sample_id, final_state, checkpoint_dir)

    result = final_state["final_result"]
    print(f"\n[Pipeline] DONE: {result.get('verdict')} | confidence={result.get('confidence', 0):.2f}")
    return result