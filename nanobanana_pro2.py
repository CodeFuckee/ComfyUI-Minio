import os
import re
import json
import time
import base64
import http.client
from pathlib import Path
from .shared import generate_random_string


class NanoBananaPro2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mimeType1": (
                    "STRING",
                    {
                        "default": "image/png",
                    },
                ),
                "imageBase641": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "mimeType2": (
                    "STRING",
                    {
                        "default": "image/png",
                    },
                ),
                "imageBase642": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "aspectRatio": (
                    "STRING",
                    {
                        "default": "9:16",
                    },
                ),
                "imageSize": (
                    "STRING",
                    {
                        "default": "2k",
                    },
                ),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"

    RETURN_TYPES = ("STRING", 'STRING',)
    RETURN_NAMES = ("response", "image",)

    def main(self, mimeType1: str, imageBase641: str, mimeType2: str, imageBase642: str, prompt: str, aspectRatio: str = '9:16', imageSize: str = '2k'):
        api_key = os.getenv("EASYART_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 EASYART_API_KEY")

        payload = json.dumps({
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{prompt}"
                        },
                        {
                            "inline_data": {
                                "mime_type": mimeType1,
                                "data": imageBase641,
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": mimeType2,
                                "data": imageBase642,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": [
                    "TEXT",
                    "IMAGE"
                ],
                "imageConfig": {
                    "aspectRatio": f"{aspectRatio}",
                    "imageSize": f"{imageSize}"
                }
            }
        })
        headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }
        max_retries = 3
        retryable_statuses = {429, 502, 503, 504}
        last_exception = None
        data = b""
        res_status = None

        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPSConnection("api.easyart.cc", timeout=6000)
                conn.request("POST", "/v1beta/models/gemini-3-pro-image-preview:generateContent", payload, headers)
                res = conn.getresponse()
                res_status = getattr(res, "status", None)
                data = res.read()

                if res_status == 200:
                    break

                if res_status in retryable_statuses and attempt < (max_retries - 1):
                    time.sleep(2 ** attempt)
                    continue

                error_text = data.decode("utf-8", errors="replace") if data else ""
                raise RuntimeError(f"请求失败，状态码: {res_status}, 响应: {error_text[:2000]}")
            except Exception as e:
                last_exception = e
                if attempt < (max_retries - 1):
                    time.sleep(2 ** attempt)
                    continue
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        decoded_data = data.decode("utf-8", errors="replace") if data else ""
        with open("response.json", "w", encoding="utf-8") as f:
            f.write(decoded_data)
        print(data)
        if res_status != 200:
            raise RuntimeError(f"请求失败，重试后仍未成功: {last_exception}")

        JSON_PATH = Path("response.json")
        OUT_DIR = Path(".")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

        parts = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
        )

        text = "\n".join(p.get("inlineData", {}).get('data', '') for p in parts if isinstance(p, dict))
        response = "".join(p.get("text") for p in parts if isinstance(p, dict) and 'text' in p)
        pattern = re.compile(r"data:image/(?P<ext>png|jpeg|jpg|webp|gif);base64,(?P<b64>[A-Za-z0-9+/=\s]+)")

        matches = list(pattern.finditer(text))
        if not matches:
            cleaned = re.sub(r"\s+", "", text)
            if not cleaned:
                raise SystemExit("没有在 JSON 里找到可用的 base64 图片数据")
            text = f"data:image/png;base64,{cleaned}"
            matches = list(pattern.finditer(text))
            if not matches:
                raise SystemExit("没有在 JSON 里找到 data:image/...;base64 的图片数据")

        for i, m in enumerate(matches, start=1):
            b64_payload = re.sub(r"\s+", "", m.group("b64"))
            img_bytes = base64.b64decode(b64_payload)
            tt = generate_random_string()
            out_path = OUT_DIR / f"extracted_image_{tt}.png"
            out_path.write_bytes(img_bytes)
            print(f"saved: {out_path.resolve()}")

        text = re.sub(r"data:image/[^;]+;base64,", "", text).strip()
        return (response, text,)
