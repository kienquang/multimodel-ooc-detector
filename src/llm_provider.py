"""
LLM Adapter — Groq API (local) ↔ vLLM 4-bit (Kaggle)

Nguyên tắc thiết kế:
  - chat_completion() là interface duy nhất mà mọi agent gọi.
  - Mọi chi tiết về Groq vs vLLM được ẩn hoàn toàn bên trong class này.
  - Structured output (Pydantic) hoạt động ở cả 2 môi trường.
  - Vision (base64 image) hoạt động ở local; Kaggle fallback sang text-only.
"""

import re
import json
from typing import Any, Optional, Type
from pydantic import BaseModel

from src.config import Config


class LLMProvider:

    def __init__(self):
        if Config.IS_KAGGLE:
            self._init_hf_local()
        else:
            self._init_groq()

    # ──────────────────────────────────────────
    # INIT
    # ──────────────────────────────────────────

    def _init_groq(self):
        from openai import OpenAI
        import instructor

        self._mode = "groq"
        self._client = OpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.BASE_URL,
        )
        # instructor wrap → structured output tự động qua Pydantic
        self._instructor = instructor.from_openai(self._client, mode=instructor.Mode.JSON)
        print(f"[LLMProvider] Groq ready | model={Config.MODEL_NAME}")

    def _init_hf_local(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

        self._mode = "hf_local"
        print(f"⏳ [LLMProvider] Đang tải {Config.MODEL_NAME} bằng Transformers (Bypass vLLM)...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # --- SỬ DỤNG BitsAndBytesConfig MỚI NHẤT ---
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # Nếu 8-bit vẫn tràn RAM, đổi thành load_in_4bit=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",     
            quantization_config=quantization_config, # <-- Thay cho load_in_8bit=True
        )
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=1024,
            temperature=Config.TEMPERATURE,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id
        )
        print(f"✅ [LLMProvider] Transformers 8-bit ready | model={Config.MODEL_NAME}")
    # ──────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────

    def chat_completion(
        self,
        messages: list[dict],
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """
        Giao diện thống nhất cho tất cả agents.

        Args:
            messages: Danh sách dict {"role": ..., "content": ...}
                      Content có thể là str hoặc list (vision).
            response_model: Pydantic model nếu cần structured output.
                           None nếu chỉ cần text thô.

        Returns:
            - Pydantic instance nếu response_model được truyền vào.
            - OpenAI response object (có .choices) nếu Groq text-only.
            - str nếu vLLM text-only.
        """
        if self._mode == "groq":
            return self._groq_completion(messages, response_model)
        else:
            return self._hf_completion(messages, response_model)
    # ──────────────────────────────────────────
    # GROQ PATH
    # ──────────────────────────────────────────

    def _groq_completion(self, messages: list[dict], response_model):
        if response_model:
            # instructor tự ép Groq trả về JSON đúng schema
            return self._instructor.chat.completions.create(
                model=Config.MODEL_NAME,
                response_model=response_model,
                messages=messages,
                temperature=Config.TEMPERATURE,
                max_retries=2,   # instructor tự retry nếu JSON sai
            )
        # Text thường → trả về OpenAI response object
        return self._client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=messages,
            temperature=Config.TEMPERATURE,
        )

    # ──────────────────────────────────────────
    # vLLM PATH
    # ──────────────────────────────────────────

    def _hf_completion(self, messages: list[dict], response_model):
        sanitized_messages = self._sanitize_messages_for_vllm(messages)
        prompt = self._tokenizer.apply_chat_template(sanitized_messages, tokenize=False, add_generation_prompt=True)
        
        # Nếu có Pydantic model, mớm schema vào prompt để ép LLM trả JSON
        if response_model:
            schema = response_model.model_json_schema()
            prompt += f"\nBạn BẮT BUỘC phải trả về kết quả dưới dạng chuỗi JSON thô, không kèm markdown, theo đúng cấu trúc sau:\n{json.dumps(schema, indent=2)}\nJSON:"

        outputs = self._pipe(prompt)
        raw_text = outputs[0]["generated_text"][len(prompt):].strip()

        if response_model:
            return self._parse_structured(raw_text, response_model)

        return _FakeResponse(raw_text)

    def _sanitize_messages_for_vllm(self, messages: list[dict]) -> list[dict]:
        """
        vLLM không hỗ trợ content dạng list (vision) — chuyển về text thuần.
        Image description sẽ được inject vào text trước đó bởi utils.py.
        """
        sanitized = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                # Lấy phần text, bỏ phần image (đã được xử lý ở utils.py)
                text_parts = [
                    part["text"] for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = " ".join(text_parts)
            sanitized.append({"role": msg["role"], "content": content})
        return sanitized

    def _parse_structured(self, text: str, response_model: Type[BaseModel]) -> BaseModel:
        """
        Parse JSON từ vLLM output một cách an toàn.

        Thứ tự thử:
          1. Parse trực tiếp (model đã trả JSON sạch)
          2. Extract JSON block bằng regex (model wrap trong markdown)
          3. model_construct() với default — KHÔNG crash pipeline
        """
        # Thử 1: parse thẳng
        try:
            return response_model.model_validate_json(text)
        except Exception:
            pass

        # Thử 2: extract JSON block từ ```json ... ``` hoặc { ... }
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not match:
            match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            try:
                return response_model.model_validate_json(match.group(1 if '```' in text else 0))
            except Exception:
                pass

        # Thử 3: parse từng field thủ công nếu JSON bị cắt
        try:
            fields = {}
            for field_name in response_model.model_fields:
                pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
                m = re.search(pattern, text)
                if m:
                    fields[field_name] = m.group(1)
            if fields:
                return response_model.model_construct(**fields)
        except Exception:
            pass

        # Fallback cuối: trả về object rỗng, LOG để debug
        print(f"⚠️  [LLMProvider] Structured parse failed. Raw:\n{text[:300]}")
        return response_model.model_construct()


class _FakeResponse:
    """
    Wrapper để vLLM text-only trả về object có .choices[0].message.content
    → Agents không cần if/else Groq vs vLLM khi đọc kết quả text.
    """
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]


class _FakeChoice:
    def __init__(self, text: str):
        self.message = _FakeMessage(text)


class _FakeMessage:
    def __init__(self, text: str):
        self.content = text


# Singleton — khởi tạo 1 lần duy nhất khi import
llm_provider = LLMProvider()