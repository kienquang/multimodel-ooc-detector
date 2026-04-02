from langgraph.graph import StateGraph, END
from typing import TypedDict, Any

from src.evidence_retriever import retrieve_evidence
from src.svo_extractor import extract_svo
from src.agents import RetrievalAgent, DetectiveAgent, AnalystAgent

class PipelineState(TypedDict):
    image_url: str
    caption: str
    evidence_list: list
    claim_svo: Any
    evidence_svo: Any
    inconsistencies: list
    deep_analysis: str
    final_result: dict          # verdict, explanation, confidence
    refine_count: int

def retrieval_node(state: PipelineState):
    evidence_list = retrieve_evidence(state["image_url"])
    claim_svo = extract_svo(state["caption"], "caption")
    evidence_svo = extract_svo(evidence_list[0]["text"], "evidence") if evidence_list else None
    retrieval = RetrievalAgent().run(claim_svo, evidence_svo)
    return {
        "evidence_list": evidence_list,
        "claim_svo": claim_svo,
        "evidence_svo": evidence_svo,
        "inconsistencies": retrieval["flagged_inconsistencies"],
        "refine_count": state.get("refine_count", 0)
    }

def detective_node(state: PipelineState):
    detective = DetectiveAgent()
    analysis = detective.run(state["inconsistencies"], state["image_url"], state["caption"])
    return {"deep_analysis": analysis["deep_analysis"]}

def analyst_node(state: PipelineState):
    analyst = AnalystAgent()
    result = analyst.run(state["deep_analysis"])
    new_refine_count = state.get("refine_count", 0) + 1
    return {
        "final_result": result,
        "refine_count": new_refine_count
    }

def should_refine(state: PipelineState):
    """Self-Refine Loop - EXCLAIM ReAct style"""
    confidence = state["final_result"].get("confidence", 0.5)
    refine_count = state.get("refine_count", 0)
    
    if confidence < 0.75 and refine_count < 3:   # MAX 3 lần refine
        print(f"🔄 Self-Refine triggered (confidence = {confidence:.2f}, refine = {refine_count})")
        return "detective"
    return END

graph = StateGraph(PipelineState)
graph.add_node("retrieval", retrieval_node)
graph.add_node("detective", detective_node)
graph.add_node("analyst", analyst_node)

graph.set_entry_point("retrieval")
graph.add_edge("retrieval", "detective")
graph.add_edge("detective", "analyst")

graph.add_conditional_edges(
    "analyst",
    should_refine,
    {"detective": "detective", END: END}
)

app = graph.compile()

def run_pipeline(image_url: str, caption: str):
    print("🚀 Starting LangGraph Multi-Agent + Self-Refine Loop (EXCLAIM + EGMMG)")
    result = app.invoke({
        "image_url": image_url,
        "caption": caption,
        "refine_count": 0
    })
    final = result["final_result"]
    print("\n🎯 FINAL RESULT (with confidence):")
    print(final)
    return final