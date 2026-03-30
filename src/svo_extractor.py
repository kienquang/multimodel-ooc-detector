
import instructor
from openai import OpenAI
from src.config import Config
from src.schemas import SVOList

instructor_client = instructor.from_openai(OpenAI(api_key=Config.GROQ_API_KEY, base_url=Config.BASE_URL))


RELATIONS = [
    "PERFORMS",       # Địa điểm (VD: Trump -> located_in -> White House)
    "LOCATED_IN",      # Thời gian (VD: Speech -> happened_on -> 2025)
    "OCCURRED_ON",  # Tham gia sự kiện
    "TARGETS",          # Trang phục (Cực kỳ quan trọng để check ảnh)
    "HAS_STATE",     # Hành động đang làm
    "SAME_AS"   # Tổ chức/Sự việc liên quan
]
def extract_svo(text: str, source: str = "caption") -> SVOList:
    print(f"📊 [SVO] Extracting from {source}...")
    system = f"""You are a strict knowledge-graph extractor (EGMMG style).
Extract ONLY real triplets that exist in the text.
Use exactly these relations: {RELATIONS}.
Never hallucinate."""
    return instructor_client.chat.completions.create(
        model=Config.MODEL_NAME,
        response_model=SVOList,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        temperature=Config.TEMPERATURE,
    )