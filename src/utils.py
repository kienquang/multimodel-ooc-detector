from openai import OpenAI
from src.config import Config
from PIL import Image
import base64
import requests

client = OpenAI(api_key=Config.GROQ_API_KEY, base_url=Config.BASE_URL)

def encode_image(image_path_or_url: str):
    if image_path_or_url.startswith(("http://", "https://")):
        # Nếu là link mạng: Tải ảnh trực tiếp vào RAM
        response = requests.get(image_path_or_url)
        response.raise_for_status() # Báo lỗi nếu link hỏng/bị chặn
        return base64.b64encode(response.content).decode('utf-8')
    else:
        # Nếu là file cục bộ: Mở từ ổ cứng
        with open(image_path_or_url, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

def get_image_description(image_url_or_path: str) -> str:
    if image_url_or_path.startswith(("http://", "https://")):
        content = [{"type": "image_url", "image_url": {"url": image_url_or_path}}]
    else:
        base64_img = encode_image(image_url_or_path)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]
    
    resp = client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe ONLY factual visual details: people, clothing, weather, location, objects, text. No speculation."},
            *content
        ]}],
        temperature=0.0
    )
    return resp.choices[0].message.content
    # print("⚠️ Bỏ qua bước nhìn ảnh (Do lỗi API Groq). Dùng mô tả mặc định.")
    # return "A photo showing Donald Trump giving a speech. He is wearing a dark suit and red tie. The background looks like the Oval Office, not an outdoor rally."