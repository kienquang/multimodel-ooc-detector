import instructor
from src.config import Config
from src.llm_provider import llm_provider   # ← DÙNG ADAPTER CHUNG
from src.schemas import SVOList

def extract_svo(text: str, source: str = "caption") -> SVOList:
    print(f"📊 [SVO] Extracting from {source}...")
    system = f"""You are a strict knowledge-graph extractor.
Extract ONLY real triplets that exist in the text.
Use exactly these relations: PERFORMS, LOCATED_IN, OCCURRED_ON, TARGETS, HAS_STATE, SAME_AS.
Never hallucinate."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text}
    ]

    return llm_provider.chat_completion(messages, response_model=SVOList)