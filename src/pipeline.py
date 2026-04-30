"""
pipeline.py — LangGraph Multi-Agent Pipeline

Thay đổi hiện tại:
  - CHỈ LOẠI BỎ S-V-O.
  - PipelineState sử dụng `claim_context` và `evidence_context` (Dạng Dictionary 6 keys).
  - Giữ nguyên toàn bộ logic cào web và Reranking cũ của Stage 1a.
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
# STATE (Đã dọn sạch S-V-O)
# ──────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    image_url:              str
    caption:                str
    image_bytes:            Optional[bytes]
    visual_entities:        list[str]
    evidence_list:          list
    claim_context:          dict  # Thay cho claim_svo
    evidence_context:       dict  # Thay cho evidence_svo
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

    # Giữ nguyên logic cào web cũ, không đụng chạm đến Reranking
    if not evidence_list:
        from src.evidence_retriever import retrieve_evidence
        evidence_list, visual_entities = retrieve_evidence(
            image_url=state["image_url"],
            image_bytes=state.get("image_bytes"),
            top_k=Config.MAX_EVIDENCE,
        )

    # 1. Trích xuất 6 Khía cạnh từ Caption (Giờ trả về Dictionary)
    claim_ctx = extract_caption_svo(state["caption"])

    # 2. Trích xuất 6 Khía cạnh từ Evidence (Giờ trả về Dictionary)
    if evidence_list or visual_entities:
        evidence_ctx = extract_contextual_svo(
            evidence_list=evidence_list,
            visual_entities=visual_entities,
            source_label="evidence",
        )
    else:
        # Fallback nếu không có bằng chứng
        evidence_ctx = {k: "Unknown" for k in ["people", "location", "date", "event", "object", "motivation"]}
        print("[Retrieval] No evidence available.")

    # 3. So sánh 1-1 để tìm mâu thuẫn (Gửi thẳng 2 dict vào RetrievalAgent mới)
    inconsistencies = RetrievalAgent().run(claim_ctx, evidence_ctx)["flagged_inconsistencies"]

    return {
        "evidence_list":          evidence_list,
        "visual_entities":        visual_entities,
        "claim_context":          claim_ctx,
        "evidence_context":       evidence_ctx,
        "inconsistencies":        inconsistencies,
        "refine_count":           state.get("refine_count", 0),
    }


def detective_node(state: PipelineState) -> dict:
    print("\n─── [Node: Detective] ───────────────────────────")
    result = DetectiveAgent().run(
        conflicts=state.get("inconsistencies", []), # Đổi thành conflicts
        image_url=state["image_url"],               # Đổi thành image_url
        caption=state["caption"]
    )
    return {"deep_analysis": result.get("deep_analysis")}


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
# CHECKPOINT & ENTRY POINT
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


def run_pipeline(
    image_url:           str,
    caption:             str,
    image_bytes:         bytes | None = None,
    sample_id:           str  = "sample",
    use_checkpoint:      bool = True,
    checkpoint_dir:      str  = "/kaggle/working/checkpoints",
    preloaded_evidence:  list | None = None,
    preloaded_entities:  list | None = None,
) -> dict:
    
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
        "evidence_list":          preloaded_evidence or [],
        "visual_entities":        preloaded_entities or [],
        "claim_context":          {}, # Đã đổi tên
        "evidence_context":       {}, # Đã đổi tên
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