import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    
    # Model chuyên đọc hiểu văn bản, băm SVO và làm Thám tử
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    BASE_URL = "https://api.groq.com/openai/v1"

    # Kaggle adapter (uncomment khi chạy Kaggle)
    # IS_KAGGLE = os.path.exists("/kaggle/input")