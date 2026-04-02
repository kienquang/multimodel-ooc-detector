from src.config import Config
from openai import OpenAI
import instructor

class LLMProvider:
    """Adapter Pattern: Groq (local) ↔ vLLM (Kaggle)"""
    
    def __init__(self):
        if Config.IS_KAGGLE:
            # vLLM 4-bit on Kaggle
            from vllm import LLM, SamplingParams
            self.is_vllm = True
            # Thay model path phù hợp với dataset bạn add trên Kaggle
            self.llm = LLM(model="/kaggle/input/llama-3-1-70b-4bit", quantization="4bit", tensor_parallel_size=2)
            self.sampling_params = SamplingParams(temperature=Config.TEMPERATURE, max_tokens=1024)
        else:
            # Groq API
            self.is_vllm = False
            self.client = OpenAI(api_key=Config.GROQ_API_KEY, base_url=Config.BASE_URL)
            self.instructor_client = instructor.from_openai(self.client)

    def chat_completion(self, messages: list, response_model=None):
        if self.is_vllm:
            # vLLM simple text generation
            prompt = "\n".join([m["content"] for m in messages])
            outputs = self.llm.generate(prompt, self.sampling_params)
            text = outputs[0].outputs[0].text
            if response_model:
                # Parse structured output (simple JSON mode)
                import json
                try:
                    return response_model.model_validate_json(text)
                except:
                    return response_model.model_validate_json(text + "}")
            return text
        else:
            # Groq
            if response_model:
                return self.instructor_client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    response_model=response_model,
                    messages=messages,
                    temperature=Config.TEMPERATURE
                )
            return self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=Config.TEMPERATURE
            )

# Singleton
llm_provider = LLMProvider()