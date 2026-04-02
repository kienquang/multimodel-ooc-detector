import base64
import requests
from io import BytesIO
from openai import OpenAI
from src.config import Config

client = OpenAI(api_key=Config.GROQ_API_KEY, base_url=Config.BASE_URL)

def encode_image(image_path_or_bytes: str | bytes) -> str:
    """Chuyển thành base64"""
    if isinstance(image_path_or_bytes, bytes):
        return base64.b64encode(image_path_or_bytes).decode()
    with open(image_path_or_bytes, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_image_description(image_url_or_path: str) -> str:
    """TRIỆT ĐỂ FIX 302: Luôn download ảnh → gửi base64"""
    print("📸 [Vision] Downloading & encoding image as base64...")

    if image_url_or_path.startswith(("http://", "https://")):
        # Download với follow redirect (max 10 lần)
        try:
            response = requests.get(
                image_url_or_path,
                allow_redirects=True,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"}   # tránh anti-hotlink
            )
            response.raise_for_status()
            image_bytes = response.content
        except Exception as e:
            print(f"⚠️  Download failed: {e}. Fallback to original URL.")
            # Fallback (ít xảy ra)
            image_bytes = None
    else:
        # Local file
        image_bytes = None

    # Chuẩn bị content cho Groq
    if image_bytes:
        base64_img = encode_image(image_bytes)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]
    else:
        # Fallback (nếu download thất bại)
        content = [{"type": "image_url", "image_url": {"url": image_url_or_path}}]

    resp = client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe ONLY factual visual details: people, clothing, weather, location, objects, text. No speculation."},
                *content
            ]
        }],
        temperature=0.0
    )
    return resp.choices[0].message.content