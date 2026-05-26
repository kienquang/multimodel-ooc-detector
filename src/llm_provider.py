"""
llm_provider.py — Dual-Core Architecture (Llama 3.1 Text + Gemma 4 Vision)

Nguyên tắc thiết kế:
  - Local (Groq API): Xử lý toàn bộ bằng API để code nhẹ, dev nhanh.
  - Kaggle (Dual-Core): 
      + GPU 0: LLaMA 3.1 8B (Xử lý Text, Trích xuất JSON, Suy luận)
      + GPU 1: Gemma 4 E4B (Thám tử thị giác) - [ĐANG TẠM KHÓA ĐỂ TEST NHANH]
"""

import os
import re
import json
import torch
from io import BytesIO
from typing import Any, Optional, Type
from pydantic import BaseModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    AutoModelForImageTextToText,   # Chuyên dụng cho Gemma 4
    AutoProcessor,
)

from src.config import Config


class LLMProvider:
    def __init__(self):
        # KHÓA HẠT GIỐNG NGẪU NHIÊN TOÀN CỤC CHO PYTORCH (Chuẩn Khoa học)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self._mode             = None
        self._text_tokenizer   = None
        self._text_pipe        = None
        self._vision_processor = None
        self._vision_model     = None

        if Config.IS_KAGGLE:
            self._init_dual_core()
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
        self._instructor = instructor.from_openai(self._client, mode=instructor.Mode.JSON)
        print(f"[LLMProvider] Groq ready | model={Config.MODEL_NAME}")

    def _init_dual_core(self):
        print("🚀 [Dual-Core] Khởi tạo hai GPU song song...")

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
            BitsAndBytesConfig,
            AutoProcessor,
            AutoModelForImageTextToText
        )

        # ── GPU 0: LLaMA 3.1 8B 8-bit — Text (Retrieval + Analyst) ──────────
        # print("⏳ [GPU 0] Loading LLaMA 3.1 8B in 8-bit...")
        # quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # self._text_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        # text_model = AutoModelForCausalLM.from_pretrained(
        #     Config.MODEL_NAME,
        #     device_map={"": 0}, 
        #     torch_dtype=torch.float16,
        #     quantization_config=quant_cfg,
        # )
        
        # # Triệt tiêu cấu hình rác gây cảnh báo
        # text_model.config.max_length = None
        # text_model.generation_config.max_length = None
        # text_model.generation_config.pad_token_id = self._text_tokenizer.eos_token_id

        # self._text_pipe = pipeline(
        #     "text-generation",
        #     model=text_model,
        #     tokenizer=self._text_tokenizer,
        #     return_full_text=False,
        # )
        # print("✅ [GPU 0] LLaMA 3.1 (8-bit) ready")

        # ── GPU 0: Gemma 4 E4B Text-Only (Tầng Text/Logic) ──────────
        print("⏳ [GPU 0] Loading Gemma 4 E4B Text-Only in 8-bit...")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # Cập nhật tên model trực tiếp hoặc qua Config.MODEL_NAME
        model_name = "principled-intelligence/gemma-4-E4B-it-text-only"

        self._text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0}, 
            torch_dtype=torch.bfloat16, # Gemma tối ưu tốt nhất trên bfloat16
            quantization_config=quant_cfg,
        )
        
        # Triệt tiêu cấu hình rác và vá lỗi pad_token của Gemma
        text_model.config.max_length = None
        text_model.generation_config.max_length = None
        if self._text_tokenizer.pad_token_id is None:
            self._text_tokenizer.pad_token = self._text_tokenizer.eos_token
            text_model.config.pad_token_id = self._text_tokenizer.eos_token_id

        self._text_pipe = pipeline(
            "text-generation",
            model=text_model,
            tokenizer=self._text_tokenizer,
            return_full_text=False,
        )
        print("✅ [GPU 0] Gemma Text-Only (8-bit) ready")
        # ── GPU 1: Gemma 4 E4B — Vision (Detective) ──────────────────────────
        # TẠM THỜI COMMENT ĐỂ TEST NHANH TEXT PIPELINE
        
        print("⏳ [GPU 1] Loading Gemma 4 E4B Vision in 4-bit...")
        vlm_id = "google/gemma-4-E4B-it"

        quant_cfg_vision = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._vision_processor = AutoProcessor.from_pretrained(vlm_id)
        self._vision_model = AutoModelForImageTextToText.from_pretrained(
            vlm_id,
            device_map={"": 1},
            torch_dtype=torch.float16,
            quantization_config=quant_cfg_vision,
            low_cpu_mem_usage=True,
        )

        self._dequantize_vision_tower()
        self._vision_model.eval()
        print("✅ [GPU 1] Gemma 4 E4B (4-bit) ready")
        
        print("⚠️ [GPU 1] ĐÃ TẠM KHÓA GEMMA 4 ĐỂ TEST NHANH THUẦN TEXT!")
        self._mode = "kaggle_dual_core"

    def _dequantize_vision_tower(self):
        """Thay Linear4bit → Linear float16 trong vision tower sau khi load."""
        import bitsandbytes as bnb
        import torch.nn as nn

        vision_tower = self._vision_model.model.vision_tower
        replaced = 0

        for module_name, module in vision_tower.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue

            parts  = module_name.split(".")
            parent = vision_tower
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]

            with torch.no_grad():
                weight_fp16 = bnb.functional.dequantize_4bit(
                    module.weight.data,
                    module.weight.quant_state,
                ).to(torch.float16)

            new_linear = nn.Linear(
                weight_fp16.shape[1],
                weight_fp16.shape[0],
                bias=module.bias is not None,
                dtype=torch.float16,
                device=module.weight.device,
            )
            new_linear.weight = nn.Parameter(weight_fp16)
            if module.bias is not None:
                new_linear.bias = nn.Parameter(
                    module.bias.data.to(torch.float16)
                )

            setattr(parent, attr, new_linear)
            replaced += 1

        print(f"✅ Vision tower: dequantized {replaced} layers → float16")

    # ──────────────────────────────────────────
    # TEXT COMPLETION (LLaMA 3.1 / GROQ)
    # ──────────────────────────────────────────

    def chat_completion(
        self,
        messages: list[dict],
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        if self._mode == "groq":
            return self._groq_completion(messages, response_model)
        else:
            return self._hf_completion(messages, response_model)

    def _groq_completion(self, messages: list[dict], response_model):
        if response_model:
            return self._instructor.chat.completions.create(
                model=Config.MODEL_NAME,
                response_model=response_model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                seed=42,
                max_retries=2,
            )
        return self._client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

    # def _hf_completion(self, messages: list[dict], response_model):
    #     sanitized_messages = self._sanitize_messages_for_vllm(messages)
    #     prompt = self._text_tokenizer.apply_chat_template(sanitized_messages, tokenize=False, add_generation_prompt=True)
        
    #     if response_model:
    #         schema = response_model.model_json_schema()
    #         prompt += f"\nBạn BẮT BUỘC phải trả về kết quả dưới dạng chuỗi JSON thô, không kèm markdown, theo đúng cấu trúc sau:\n{json.dumps(schema, indent=2)}\nJSON:\n"

    #     outputs = self._text_pipe(
    #         prompt,
    #         max_new_tokens=1024,
    #         # max_length=None, 
    #         do_sample=False,
    #     )
        
    #     generated_text = outputs[0]["generated_text"]

    #     if generated_text.startswith(prompt):
    #         raw_text = generated_text[len(prompt):].strip()
    #     else:
    #         raw_text = generated_text.strip()

    #     if response_model:
    #         return self._parse_structured(raw_text, response_model)

    #     return _FakeResponse(raw_text)

    def _hf_completion(self, messages: list[dict], response_model):
        
        # 1. BƠM LỆNH ÉP JSON VÀO TIN NHẮN CUỐI CÙNG CỦA USER (TRƯỚC KHI TEMPLATE HÓA)
        if response_model:
            schema = response_model.model_json_schema()
            # Biến Schema phức tạp thành Template đơn giản cho 7B dễ hiểu
            template_dict = {k: f"<{v.get('type', 'string')}>" for k, v in schema.get('properties', {}).items()}
            
            json_instruction = (
                "\n\n--- FORMAT INSTRUCTION ---\n"
                "You MUST return ONLY a valid JSON object. Do NOT wrap it in markdown blocks (```json). "
                "Do NOT output the schema definitions. Output the ACTUAL DATA matching this exact template:\n"
                f"{json.dumps(template_dict, indent=2)}"
            )
            
            # Gắn lệnh này vào nội dung của User
            for msg in reversed(messages):
                if msg["role"] == "user":
                    if isinstance(msg["content"], list):
                        msg["content"].append({"type": "text", "text": json_instruction})
                    else:
                        msg["content"] += json_instruction
                    break

        # 2. Gom role System vào User (Đặc trị cho Gemma)
        sanitized_messages = self._sanitize_messages_for_vllm(messages)
        
        # 3. Áp dụng template chuẩn của Mô hình
        prompt = self._text_tokenizer.apply_chat_template(
            sanitized_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 4. Chạy model
        outputs = self._text_pipe(
            prompt,
            max_new_tokens=1024,
            max_length=8192, 
            do_sample=False,
        )
        
        generated_text = outputs[0]["generated_text"]

        # Cắt bỏ phần prompt đầu vào (vì pipeline đôi khi trả về cả prompt)
        if generated_text.startswith(prompt):
            raw_text = generated_text[len(prompt):].strip()
        else:
            raw_text = generated_text.strip()

        if response_model:
            return self._parse_structured(raw_text, response_model)

        return _FakeResponse(raw_text)
    # ──────────────────────────────────────────
    # VISION COMPLETION (GEMMA 4 E4B)
    # ──────────────────────────────────────────

    def vision_completion(self, image_source: Any, prompt_text: str) -> str:
        from PIL import Image
        import requests

        if self._mode != "kaggle_dual_core":
            return self._groq_vision_fallback(image_source, prompt_text)

        # LƯỚI AN TOÀN: Bypass nếu Gemma bị comment
        if self._vision_model is None or self._vision_processor is None:
            print("⚠️ [Vision] Bỏ qua xử lý ảnh vì Gemma 4 đang bị tạm khóa!")
            return "Vision Analysis Bypassed: Gemma 4 đang được tạm khóa để test nhanh."

        if isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source)).convert("RGB")
        elif str(image_source).startswith(("http://", "https://")):
            raw = requests.get(image_source, timeout=10).content
            image = Image.open(BytesIO(raw)).convert("RGB")
        else:
            image = Image.open(image_source).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image",  "image": image},
                {"type": "text",   "text": prompt_text},
            ]
        }]

        inputs = self._vision_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._vision_model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._vision_model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        input_len   = inputs["input_ids"].shape[-1]
        new_tokens  = output_ids[:, input_len:]
        return self._vision_processor.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

    def _groq_vision_fallback(self, image_source: Any, prompt_text: str) -> str:
        import base64
        from groq import Groq
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        if isinstance(image_source, bytes):
            b64_str = base64.b64encode(image_source).decode("utf-8")
        else:
            with open(image_source, "rb") as img_file:
                b64_str = base64.b64encode(img_file.read()).decode('utf-8')

        resp = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
                ]
            }],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )
        return resp.choices[0].message.content

    # ──────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────

    # def _sanitize_messages_for_vllm(self, messages: list[dict]) -> list[dict]:
    #     sanitized = []
    #     for msg in messages:
    #         content = msg["content"]
    #         if isinstance(content, list):
    #             text_parts = [
    #                 part["text"] for part in content
    #                 if isinstance(part, dict) and part.get("type") == "text"
    #             ]
    #             content = " ".join(text_parts)
    #         sanitized.append({"role": msg["role"], "content": content})
    #     return sanitized

    def _sanitize_messages_for_vllm(self, messages: list[dict]) -> list[dict]:
        sanitized = []
        system_prompt = ""

        for msg in messages:
            content = msg["content"]
            # Bóc tách nội dung nếu là dạng list
            if isinstance(content, list):
                text_parts = [
                    part["text"] for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = " ".join(text_parts)
                
            # ĐẶC TRỊ CHO GEMMA: Gom role 'system' lại
            if msg["role"] == "system":
                system_prompt += content + "\n\n"
            else:
                # Nếu là role user đầu tiên, dán system prompt lên đầu
                if msg["role"] == "user" and system_prompt:
                    content = system_prompt + content
                    system_prompt = "" # Xóa sau khi đã dán
                    
                sanitized.append({"role": msg["role"], "content": content})
                
        return sanitized
    
    def _parse_structured(self, text: str, response_model: Type[BaseModel]) -> BaseModel:
        try:
            return response_model.model_validate_json(text)
        except Exception:
            pass

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not match:
            match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            try:
                return response_model.model_validate_json(match.group(1 if '```' in text else 0))
            except Exception:
                pass

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

        print(f"⚠️  [LLMProvider] Structured parse failed. Raw:\n{text[:300]}")
        return response_model.model_construct()

class _FakeResponse:
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]

class _FakeChoice:
    def __init__(self, text: str):
        self.message = _FakeMessage(text)

class _FakeMessage:
    def __init__(self, text: str):
        self.content = text

llm_provider = LLMProvider()