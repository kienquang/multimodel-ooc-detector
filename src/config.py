import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ============== COMMON ==============
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

    # ============== ENVIRONMENT DETECTION ==============
    IS_KAGGLE = os.path.exists("/kaggle/input")
    
    if IS_KAGGLE:
        print("🚀 Detected Kaggle environment → Using vLLM 4-bit + CLIP batch")
        MODEL_NAME = "local-vllm"   # placeholder
        BASE_URL = None
    else:
        print("💻 Local laptop → Using Groq API")
        MODEL_NAME = os.getenv("MODEL_NAME", "llama-4-scout-17b-16e-instruct")
        BASE_URL = "https://api.groq.com/openai/v1"