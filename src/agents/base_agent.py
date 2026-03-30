from openai import OpenAI
from src.config import Config

class BaseAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.GROQ_API_KEY, base_url=Config.BASE_URL)