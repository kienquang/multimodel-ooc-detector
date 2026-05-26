"""
pipeline.py — LangGraph Multi-Agent Pipeline (V5: 4-Layer Architecture)

Luồng xử lý:
    [Tầng 1] Retrieval Node  → Nạp evidence từ Cache (ĐÃ ĐƯỢC RERANK BỞI JINA-CLIP TỪ TRƯỚC).
    [Tầng 2] Detective Node  → EvidenceCoordinator (Temporal + Semantic agents).
    [Tầng 3] Visual Node     → Gemma 4 visual grounding (khi INSUFFICIENT hoặc NO_EVIDENCE).
    [Tầng 4] Analyst Node    → Map verdict sang final output (không gọi LLM nếu có evidence).

Số LLM calls theo path:
    temporal_short_circuit      → 1  call  (Tầng 2)
    semantic_no_contradiction   → 2  calls (Tầng 2)
    semantic_contradiction      → 2  calls (Tầng 2)
    insufficient + visual       → 2 + Gemma calls (Tầng 2 + 3)
    no_evidence (internal mode) → Gemma only (Tầng 3)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.analyst_agent import AnalystAgent
from src.agents.evidence_coordinator import EvidenceCoordinator
from src.config import Config
import re

# ──────────────────────────────────────────────────────────────────────────── #
# STATE
# ──────────────────────────────────────────────────────────────────────────── #

class PipelineState(TypedDict):
    # --- Input ---
    image_url:          str
    caption:            str
    image_bytes:        Optional[bytes]

    # --- Tầng 1: Retrieval ---
    visual_entities:    List[str]
    evidence_list:      List[Dict]
    evidence_context:   Dict          # raw_text gộp top-2 articles (dùng bởi RetrievalAgent)

    # --- Tầng 2: Text Agents ---
    coordinator_result: Dict          # Output của EvidenceCoordinator

    # --- Tầng 3: Visual Grounding ---
    needs_visual:       bool          # True khi INSUFFICIENT hoặc NOT_ENOUGH_INFO
    visual_mode:        str           # "grounding" | "internal" | "none"
    deep_analysis:      str           # Raw Gemma output (chỉ khi needs_visual=True)

    # --- Tầng 4: Analyst ---
    final_result:       Dict


# ──────────────────────────────────────────────────────────────────────────── #
# NODE 1 — Retrieval (Cache Mode - Top 1 Strictly)
# ──────────────────────────────────────────────────────────────────────────── #

def retrieval_node(state: PipelineState) -> Dict:
    print("\n─── [Node: Retrieval / Tầng 1] ─────────────────────────────")

    evidence_list   = state.get("evidence_list", [])
    visual_entities = state.get("visual_entities", [])

    if not evidence_list:
        print("⚠️ [Retrieval] Cảnh báo: Không có evidence từ cache. Đang chạy Fallback mode...")
        try:
            from src.evidence_retriever import retrieve_evidence
            evidence_list, visual_entities = retrieve_evidence(
                image_url   = state["image_url"],
                image_bytes = state.get("image_bytes"),
                top_k       = Config.MAX_EVIDENCE,
            )
        except ImportError:
            print("❌ [Retrieval] Fallback thất bại: Không tìm thấy module retrieve_evidence.")

    print(f"[Retrieval] Đã tiếp nhận {len(evidence_list)} evidence (Đã được Rerank bởi Hệ thống Phân tầng).")

    # Không còn đóng gói evidence_context hay gọi RetrievalAgent nữa.
    # Tầng 2 (Detective Node) sẽ trực tiếp xử lý mảng evidence_list.
    
    return {
        "evidence_list":    evidence_list,
        "visual_entities":  visual_entities,
        "evidence_context": {}, # Xóa sạch
        "_inconsistencies": [], # Xóa sạch
    }


# ──────────────────────────────────────────────────────────────────────────── #
# NODE 2 — Detective (Tầng 2: EvidenceCoordinator)
# ──────────────────────────────────────────────────────────────────────────── #

def detective_node(state: PipelineState) -> Dict:
    print("\n─── [Node: Detective / Tầng 2] ─────────────────────────────")

    coordinator = EvidenceCoordinator()
    result      = coordinator.run(
        caption       = state["caption"],
        evidence_list = state.get("evidence_list", []),
    )

    verdict = result.get("verdict")

    # Xác định xem Tầng 3 có cần chạy không
    needs_visual = verdict in ("INSUFFICIENT", "NOT_ENOUGH_INFO")
    visual_mode  = (
        "internal"   if verdict == "NOT_ENOUGH_INFO"
        else "grounding" if verdict == "INSUFFICIENT"
        else "none"
    )

    print(f"[Detective] coordinator_verdict={verdict} | needs_visual={needs_visual} | visual_mode={visual_mode}")

    return {
        "coordinator_result": result,
        "needs_visual":       needs_visual,
        "visual_mode":        visual_mode,
        "deep_analysis":      "",   # Sẽ được điền bởi visual_node nếu cần
    }


# ──────────────────────────────────────────────────────────────────────────── #
# NODE 3 — Visual Grounding (Tầng 3: Gemma 4)
# ──────────────────────────────────────────────────────────────────────────── #

def visual_node(state: PipelineState) -> Dict:
    print("\n─── [Node: Visual / Tầng 3] ─────────────────────────────────")

    visual_mode = state.get("visual_mode", "none")

    if visual_mode == "none":
        print("[Visual] Không cần visual grounding → skip")
        return {"deep_analysis": ""}

    from src.agents.detective_agent import DetectiveAgent
    caption       = state["caption"]
    image_url     = state["image_url"]
    
    evidence_list = state.get("evidence_list", [])
    evidence_text = evidence_list[0].get("optimized_evidence", "") if evidence_list else ""

    # KHAI BÁO SẴN PROMPT INTERNAL ĐỂ DÙNG CHUNG KHI CẦN FALLBACK
    prompt_internal = (
        "[INTERNAL_MODE] You are a strict visual caption verifier.\n\n"

        "TASK\n"
        "Determine whether the CAPTION's core visual claim is affirmatively "
        "supported by what is directly observable in the image.\n\n"

        "VERDICT OPTIONS\n"
        "NO_CONTRADICTION\n"
        "  The image explicitly shows the core subjects, actions, and scene "
        "type described in the caption. The visual evidence actively aligns "
        "with the caption's main visual claim.\n\n"
        
        "CONTRADICTION\n"
        "  Use this verdict if the image fails to actively confirm the caption's "
        "core visual claim. This includes:\n"
        "  (i) The image shows a completely different subject, scene, or event.\n"
        "  (ii) The core objects or actions described in the caption are absent from the image.\n"
        "  (iii) The image visually conflicts with the caption's description.\n\n"

        "WHAT TO IGNORE (INVISIBLE METADATA)\n"
        "You must still ignore unobservable details (dates, specific names, geographic locations, internal motives). "
        "DO NOT output CONTRADICTION just because you cannot verify a specific name or date visually. "
        "Focus ONLY on whether the fundamental visual action/scene matches.\n\n"

        "CRITICAL RULE: STRICT VISUAL MATCHING\n"
        "The burden of proof is on the caption. If the caption claims a specific visual event "
        "(e.g., 'a violent protest', 'abandoned green scooters'), the image MUST show that event. "
        "If the image merely shows a generic scene lacking those core visual elements, "
        "you MUST output CONTRADICTION. There is no INSUFFICIENT verdict.\n\n"

        "--------------------------------------------------\n"
        f"CAPTION TO VERIFY: {caption}\n\n"

        "Evaluate step by step based only on what you can directly observe in the image:\n"
        "Step 1 — Identify the caption's core visual claim. Strip out invisible metadata (names, dates, invisible causes).\n"
        "Step 2 — Describe the actual visual content of the image.\n"
        "Step 3 — Does the visual content in Step 2 affirmatively match the core visual claim in Step 1? "
        "If the core visual elements are missing or different, verdict is CONTRADICTION. "
        "If they clearly match, verdict is NO_CONTRADICTION.\n\n"

        "--- OUTPUT FORMAT ---\n"
        "You MUST structure your response EXACTLY like this:\n"
        "REASONING: <Write your steps 1-3 here>\n"
        "VERDICT: <Output exactly [NO_CONTRADICTION] or [CONTRADICTION]>"
    )

    if visual_mode == "grounding":
        print("[Visual] Chế độ GROUNDING — Gemma kiểm tra: evidence có mô tả đúng ảnh gốc không?")

        prompt_grounding = (
            "[GROUNDING_MODE] You are an Image-Source topical matching expert.\n\n"
            
            "CONTEXT\n"
            "The EVIDENCE passages are crawled from the web. IGNORE THE CAPTION FOR NOW. "
            "Your ONLY job is to compare the IMAGE against the provided EVIDENCE to see "
            "if they share the same core subject matter.\n\n"
            
            "TASK\n"
            "Determine if the EVIDENCE text is topically related to the visual "
            "contents of the IMAGE. You are checking for OVERLAP, not a perfect description.\n\n"
            
            "VERDICT OPTIONS\n"
            "MATCHES\n"
            "  The evidence mentions or discusses the core objects, people, or general subjects "
            "visible in the image (e.g., if the image shows scooters, and the evidence discusses scooters). "
            "The evidence DOES NOT need to explicitly describe the specific scene or action. "
            "Sharing the main visual subject matter is enough to constitute a match.\n\n"
            
            "DOES_NOT_MATCH\n"
            "  The evidence is completely unrelated to the visual contents of the image "
            "and shares no core objects or subjects.\n\n"
            
            "CRITICAL RULE\n"
            "There is no INSUFFICIENT verdict. You must choose MATCHES if there is ANY topical "
            "overlap between the text and the image objects. Choose DOES_NOT_MATCH ONLY if they "
            "are completely disjointed.\n\n"
            
            "--------------------------------------------------\n"
            f"EVIDENCE TO VERIFY:\n{evidence_text[:1500]}\n\n"
            
            "Evaluate step by step:\n"
            "Step 1 — Identify the core objects, people, or elements visible in the image.\n"
            "Step 2 — Check if the evidence text mentions those same core elements. "
            "Remember: partial topical overlap is sufficient for a MATCH.\n"
            "Step 3 — Conclude whether they share subject matter.\n\n"
            
            "--- OUTPUT FORMAT ---\n"
            "You MUST structure your response EXACTLY like this:\n"
            "REASONING: <Write your steps 1-3 here>\n"
            "VERDICT: <Output exactly [MATCHES] or [DOES_NOT_MATCH]>"
        )
        
        analysis_result = DetectiveAgent().run(image_url=image_url, prompt=prompt_grounding)
        deep_analysis = analysis_result.get("deep_analysis", "")
        
        # 🚀 CƠ CHẾ FALLBACK (Đã sửa lỗi ngoặc vuông) 🚀
        upper_analysis = deep_analysis.upper()
        # Chỉ cần tìm chữ, không cần quan tâm ngoặc vuông
        if "DOES_NOT_MATCH" in upper_analysis or "INSUFFICIENT" in upper_analysis:
            print("\n⚠️ [Visual] Evidence rác! Kích hoạt chiến dịch FALLBACK sang chế độ INTERNAL...")
            analysis_result_fallback = DetectiveAgent().run(image_url=image_url, prompt=prompt_internal)
            
            # Ghi đè kết quả bằng báo cáo của chế độ Internal
            deep_analysis = analysis_result_fallback.get("deep_analysis", "")
            deep_analysis += "\n\n[SYSTEM_NOTE: This result was generated via INTERNAL FALLBACK]"
            
    else:  # internal
        print("[Visual] Chế độ INTERNAL — Gemma verify caption vs ảnh gốc trực tiếp")
        analysis_result = DetectiveAgent().run(image_url=image_url, prompt=prompt_internal)
        deep_analysis = analysis_result.get("deep_analysis", "")

    return {"deep_analysis": deep_analysis}


# ──────────────────────────────────────────────────────────────────────────── #
# NODE 4 — Analyst (Tầng 4: Final verdict mapping)
# ──────────────────────────────────────────────────────────────────────────── #

_COORDINATOR_VERDICT_MAP: Dict[str, str] = {
    "CONTRADICTION":   "FAKE",
    "NO_CONTRADICTION": "TRUE",
}

def analyst_node(state: PipelineState) -> Dict:
    print("\n─── [Node: Analyst / Tầng 4] ────────────────────────────────")

    coordinator_result = state.get("coordinator_result", {})
    coordinator_verdict = coordinator_result.get("verdict", "INSUFFICIENT")
    needs_visual = state.get("needs_visual", False)
    visual_mode  = state.get("visual_mode", "none")

    # ── Nhánh evidence: direct mapping ────────────── #
    if not needs_visual and coordinator_verdict in _COORDINATOR_VERDICT_MAP:
        final_verdict = _COORDINATOR_VERDICT_MAP[coordinator_verdict]
        path          = coordinator_result.get("path", "direct_map")
        confidence    = coordinator_result.get("confidence", "HIGH")
        llm_calls     = coordinator_result.get("llm_calls", 0)

        print(f"[Analyst] Nhánh EVIDENCE — direct map: {coordinator_verdict} → {final_verdict}")
        print(f"[Analyst] path={path} | confidence={confidence} | total_llm_calls={llm_calls}")

        return {
            "final_result": {
                "verdict":         final_verdict,
                "path":            path,
                "confidence":      confidence,
                "llm_calls_total": llm_calls,
                "coordinator":     coordinator_result,
                "visual":          None,
            }
        }

    # ── Nhánh visual: BÓC TÁCH KẾT QUẢ TỪ GEMMA (KHÔNG GỌI LLM NỮA) ────────── #
    deep_analysis = state.get("deep_analysis", "")
    llm_calls_text = coordinator_result.get("llm_calls", 0)

    print(f"[Analyst] Nhánh VISUAL ({visual_mode}) — Dùng Regex bóc tách kết quả từ Gemma")

    final_verdict = "NOT_ENOUGH_INFO"
    upper_analysis = deep_analysis.upper()
    
    # 1. Tìm các từ khóa trong ngoặc vuông (Bao gồm cả MATCHES và DOES_NOT_MATCH)
    match = re.search(r'\[(MATCHES|DOES_NOT_MATCH|FAKE|TRUE|CONTRADICTION|NO_CONTRADICTION|INSUFFICIENT)\]', upper_analysis)
    
    if match:
        raw_verdict = match.group(1)
    else:
        # Fallback tìm 100 ký tự cuối
        tail = upper_analysis[-100:]
        if "MATCHES" in tail and "DOES_NOT_MATCH" not in tail:
            raw_verdict = "MATCHES"
        elif "DOES_NOT_MATCH" in tail:
            raw_verdict = "DOES_NOT_MATCH"
        elif "FAKE" in tail or "CONTRADICTION" in tail and "NO_CONTRADICTION" not in tail:
            raw_verdict = "FAKE"
        elif "TRUE" in tail or "NO_CONTRADICTION" in tail:
            raw_verdict = "TRUE"
        elif "INSUFFICIENT" in tail:
            raw_verdict = "INSUFFICIENT"
        else:
            raw_verdict = "UNKNOWN"

    # 2. Ánh xạ từ khóa về nhãn chuẩn của hệ thống
    if raw_verdict in ["FAKE", "CONTRADICTION", "MATCHES"]:
        final_verdict = "FAKE"
    elif raw_verdict in ["TRUE", "NO_CONTRADICTION"]:
        final_verdict = "TRUE"
    elif raw_verdict in ["INSUFFICIENT", "DOES_NOT_MATCH"]:
        final_verdict = "NOT_ENOUGH_INFO"

    print(f"[Analyst] Gemma verdict (Regex Parsed): {final_verdict} (Raw: {raw_verdict})")
    print(f"[Analyst] path=visual_{visual_mode} | total_llm_calls={llm_calls_text}+Gemma")
    return {
        "final_result": {
            "verdict":         final_verdict,
            "path":            f"visual_{visual_mode}",
            "confidence":      "MEDIUM", 
            "llm_calls_total": f"{llm_calls_text} text + 1 Vision",
            "coordinator":     coordinator_result,
            "visual":          {"raw_analysis": deep_analysis, "verdict": final_verdict},
        }
    }


# ──────────────────────────────────────────────────────────────────────────── #
# ROUTING: needs_visual?
# ──────────────────────────────────────────────────────────────────────────── #

def _route_after_detective(state: PipelineState) -> str:
    """Detective → visual_node nếu cần, ngược lại → analyst_node."""
    if state.get("needs_visual", False):
        return "visual"
    return "analyst"


# ──────────────────────────────────────────────────────────────────────────── #
# GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────── #

def _build_graph() -> Any:
    graph = StateGraph(PipelineState)

    graph.add_node("retrieval", retrieval_node)
    graph.add_node("detective", detective_node)
    graph.add_node("visual",    visual_node)
    graph.add_node("analyst",   analyst_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "detective")

    # Conditional routing sau Detective
    graph.add_conditional_edges(
        "detective",
        _route_after_detective,
        {
            "visual":  "visual",
            "analyst": "analyst",
        },
    )

    graph.add_edge("visual",  "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()


_app = _build_graph()


# ──────────────────────────────────────────────────────────────────────────── #
# CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────── #

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


# ──────────────────────────────────────────────────────────────────────────── #
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────── #

def run_pipeline(
    image_url:          str,
    caption:            str,
    image_bytes:        Optional[bytes] = None,
    sample_id:          str             = "sample",
    use_checkpoint:     bool            = True,
    checkpoint_dir:     str             = "/kaggle/working/checkpoints",
    preloaded_evidence: Optional[List]  = None,
    preloaded_entities: Optional[List]  = None,
) -> dict:

    Config.log_env()
    Config.validate()

    if use_checkpoint and Config.IS_KAGGLE:
        cached = _load_checkpoint(sample_id, checkpoint_dir)
        if cached and "final_result" in cached:
            return cached["final_result"]

    mode = "batch (preloaded)" if preloaded_evidence is not None else "demo (SerpAPI)"
    print(f"\n[Pipeline] START | id={sample_id} | mode={mode}")
    print(f"  Image  : {str(image_url)[:80]}")
    print(f"  Caption: {caption[:100]}")

    initial_state: PipelineState = {
        "image_url":          image_url,
        "caption":            caption,
        "image_bytes":        image_bytes,
        "visual_entities":    preloaded_entities or [],
        "evidence_list":      preloaded_evidence or [],
        "evidence_context":   {},
        "coordinator_result": {},
        "needs_visual":       False,
        "visual_mode":        "none",
        "deep_analysis":      "",
        "final_result":       {},
    }

    final_state = _app.invoke(initial_state)

    if Config.IS_KAGGLE:
        _save_checkpoint(sample_id, final_state, checkpoint_dir)

    result = final_state["final_result"]
    print(
        f"\n[Pipeline] DONE: verdict={result.get('verdict')} | "
        f"path={result.get('path')} | llm_calls={result.get('llm_calls_total')}"
    )
    return result