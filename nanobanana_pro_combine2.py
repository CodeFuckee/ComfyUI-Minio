import os
import re
import cv2
import json
import time
import torch
import base64
import secrets
import http.client
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO


class NanoBananaProCombine2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line": (["fast", "stable", "economy", "cheap"],),
                "model": (["nano banana 2", "nano banana pro"],),
                "mimeType": (
                    "STRING",
                    {
                        "default": "image/png",
                    },
                ),
                "imageBase64": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "imageBase64_1": (
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

    def get_header(self, line: str = 'cheap') -> dict:
        api_key_dict = {
            "fast": os.getenv("EASYART_FAST_API_KEY"),
            "stable": os.getenv("EASYART_STABLE_API_KEY"),
            "economy": os.getenv("EASYART_API_KEY"),
            "cheap": os.getenv("GRSAI_API_KEY")
        }
        api_key = api_key_dict[line]
        return {
            'Authorization': f'Bearer {api_key}' if line == 'cheap' else api_key,
            'Content-Type': 'application/json'
        }

    def get_api_host(self, line: str = 'cheap') -> str:
        if line == 'cheap':
            return 'grsai.dakka.com.cn'
        else:
            return 'api.easyart.cc'

    def get_model_name(self, line: str = 'cheap', model: str = 'nano banana pro') -> str:
        model_dict = {
            "fast": {
                'nano banana 2': 'gemini-3.1-flash-image-preview',
                'nano banana pro': 'gemini-3-pro-image-preview',
                'nano banana': 'gemini-2.5-flash-image',
            },
            "stable": {
                'nano banana 2': 'gemini-3.1-flash-image-preview',
                'nano banana pro': 'gemini-3-pro-image-preview',
                'nano banana': 'gemini-2.5-flash-image',
            },
            "economy": {
                'nano banana 2': 'gemini-3.1-flash-image-preview',
                'nano banana pro': 'gemini-3-pro-image-preview',
                'nano banana': 'gemini-2.5-flash-image',
            },
            "cheap": {
                'nano banana 2': 'nano-banana-2',
                'nano banana pro': 'nano-banana-pro',
                'nano banana': 'nano-banana-fast',
            },
        }
        model = model.lower()
        return model_dict[line][model]

    def handle_response(self, decoded_data: str):
        data = json.loads(decoded_data)
        if 'candidates' in data and isinstance(data['candidates'], list):
            candidates = data['candidates']
            for candidate in candidates:
                if 'content' not in candidate:
                    continue
                content = candidate['content']
                if 'parts' not in content or not isinstance(content['parts'], list):
                    continue
                parts = content['parts']
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if 'text' in part:
                        s = part['text']
                        pattern = r'https?://[^\s)]+'
                        match = re.search(pattern, s)
                        if match:
                            img_url = match.group()
                            # 下载图片URL并转为base64
                            import urllib.request
                            with urllib.request.urlopen(img_url) as resp:
                                img_data = resp.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            return (decoded_data, img_base64,)
                    if 'inlineData' in part:
                        inlineData = part['inlineData']
                        if 'data' not in inlineData:
                            continue
                        text ="\n".join(str(inlineData['data']))
                        pattern = re.compile(r"data:image/(?P<ext>png|jpeg|jpg|webp|gif);base64,(?P<b64>[A-Za-z0-9+/=\s]+)")
                        matches = list(pattern.finditer(text))
                        if not matches:
                            cleaned = re.sub(r"\s+", "", text)
                            if not cleaned:
                                continue
                            text = f"data:image/png;base64,{cleaned}"
                            matches = list(pattern.finditer(text))
                            if not matches:
                                continue
                        text = re.sub(r"data:image/[^;]+;base64,", "", text).strip()
                        return (decoded_data, text,)
        ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
        rand6 = f"{secrets.randbelow(1000000):06d}"
        debug_name = f"{ts}{rand6}.json"
        Path(debug_name).write_text(
            decoded_data,
            encoding="utf-8",
        )
        raise ValueError(f"no img debug file {debug_name}")

    def main(self, line: str, model: str, mimeType: str, imageBase64: str, imageBase64_1: str, prompt: str, aspectRatio: str = '9:16', imageSize: str = '2k'):
        parts = [
            {
                "text": f"{prompt}"
            },
        ]
        if imageBase64 != '-1' and imageBase64 != '':
            parts.append({
                "inline_data": {
                    "mime_type": mimeType,
                    "data": imageBase64,
                }
            })
        if imageBase64_1 != '-1' and imageBase64_1 != '':
            parts.append({
                "inline_data": {
                    "mime_type": mimeType,
                    "data": imageBase64_1,
                }
            })
        payload_json = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
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
        }
        payload = json.dumps(payload_json)
        max_retries = 1
        retryable_statuses = {429, 502, 503, 504}
        last_exception = None
        data = b""
        res_status = None

        for attempt in range(max_retries):
            try:
                print("第一次尝试请求：", attempt == 0)
                api_host = self.get_api_host(line)
                headers = self.get_header(line)
                model_name = self.get_model_name(line, model)
                conn = http.client.HTTPSConnection(api_host, timeout=600)
                conn.request("POST", f"/v1beta/models/{model_name}:generateContent", payload, headers)
                res = conn.getresponse()
                res_status = getattr(res, "status", None)
                data = res.read()
                decoded_data_test = data.decode("utf-8", errors="replace") if data else ""
                if res_status == 200 and decoded_data_test != "":
                    break
                print(f'状态码{res_status}')
                if res_status in retryable_statuses and attempt < (max_retries - 1):
                    time.sleep(2 ** attempt)
                    continue
                error_text = data.decode("utf-8", errors="replace") if data else ""
                raise RuntimeError(f"请求失败，状态码: {res_status}, 响应: {error_text[:2000]}")
            except Exception as e:
                print('请求失败' + str(e))
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
        if res_status != 200:
            raise RuntimeError(f"请求失败，重试后仍未成功: {last_exception}")
        return self.handle_response(decoded_data)

# test = NanoBananaProCombine2()
# a = ''
# b = ''
# with open('response1.json') as f:
#     text = f.read()
#     a1, a2 = test.handle_response(text)
#     # 将b2（torch张量）保存为PNG图片
#     # b2形状为 (1, H, W, 3)，值域 [0,1]
#     # img_tensor = a2.squeeze(0)          # 去掉batch维度 -> (H, W, 3)
#     # img_np = (img_tensor.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
#     # img_pil = Image.fromarray(img_np)
#     # img_pil.save("output_a2.png")
#     try:
#         img_bytes = base64.b64decode(a2, validate=True)
#     except Exception:
#         img_bytes = base64.urlsafe_b64decode(text + "===")
#     img = Image.open(BytesIO(img_bytes))
#     img.load()
#     buf = BytesIO()
#     img.save("output_a2.png")
# with open('response2.json') as f:
#     text = f.read()
#     b1, b2 = test.handle_response(text)
#     # 将b2（torch张量）保存为PNG图片
#     # b2形状为 (1, H, W, 3)，值域 [0,1]
#     # img_tensor = b2.squeeze(0)          # 去掉batch维度 -> (H, W, 3)
#     # img_np = (img_tensor.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
#     # img_pil = Image.fromarray(img_np)
#     # img_pil.save("output_b2.png")
#     try:
#         img_bytes = base64.b64decode(b2, validate=True)
#     except Exception:
#         img_bytes = base64.urlsafe_b64decode(text + "===")
#     img = Image.open(BytesIO(img_bytes))
#     img.load()
#     buf = BytesIO()
#     img.save("output_b2.png")
# print('')
