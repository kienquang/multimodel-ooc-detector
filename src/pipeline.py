from langgraph.graph import StateGraph, END
from typing import TypedDict
from src.evidence_retriever import retrieve_evidence
from src.svo_extractor import extract_svo
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.detective_agent import DetectiveAgent
from src.agents.analyst_agent import AnalystAgent
from src.utils import get_image_description

class PipelineState(TypedDict):
    image_url: str
    caption: str
    evidence_list: list
    claim_svo: object
    evidence_svo: object
    inconsistencies: list
    deep_analysis: str
    final_result: str

def retrieval_node(state: PipelineState):
    evidence_list = retrieve_evidence(state["image_url"])
    claim_svo = extract_svo(state["caption"], "caption")
    evidence_svo = extract_svo(evidence_list[0]["text"], "evidence") if evidence_list else None
    retrieval = RetrievalAgent().run(claim_svo, evidence_svo)
    return {"evidence_list": evidence_list, "claim_svo": claim_svo, "evidence_svo": evidence_svo, "inconsistencies": retrieval["flagged_inconsistencies"]}

def detective_node(state: PipelineState):
    detective = DetectiveAgent()
    analysis = detective.run(state["inconsistencies"], state["image_url"], state["caption"])
    return {"deep_analysis": analysis["deep_analysis"]}

def analyst_node(state: PipelineState):
    analyst = AnalystAgent()
    result = analyst.run(state["deep_analysis"])
    return {"final_result": result}

graph = StateGraph(PipelineState)
graph.add_node("retrieval", retrieval_node)
graph.add_node("detective", detective_node)
graph.add_node("analyst", analyst_node)

graph.set_entry_point("retrieval")
graph.add_edge("retrieval", "detective")
graph.add_edge("detective", "analyst")
graph.add_edge("analyst", END)

app = graph.compile()

def run_pipeline(image_url: str, caption: str):
    print("🚀 Starting LangGraph Multi-Agent Pipeline (EXCLAIM + EGMMG inspired)")
    result = app.invoke({"image_url": image_url, "caption": caption})
    print("\n🎯 FINAL RESULT:")
    print(result["final_result"])
    return result["final_result"]