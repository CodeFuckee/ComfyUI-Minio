import os
import re
import json
import time
import http.client


class NanoBananaProCombine:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_source": (["grsai", "modelhub"],),
                "model": (["nano banana 2", "nano banana pro", "nano banana pro vt", "nano banana"],),
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

    def get_header(self, api_source: str = 'modelhub') -> dict:
        if api_source == 'grsai':
            api_key = os.getenv("GRSAI_API_KEY")
            if not api_key:
                raise RuntimeError("缺少环境变量 GRSAI_API_KEY")
        else:
            api_key = os.getenv("EASYART_API_KEY")
            if not api_key:
                raise RuntimeError("缺少环境变量 EASYART_API_KEY")
        return {
            'Authorization': f'Bearer {api_key}' if api_source == 'grsai' else api_key,
            'Content-Type': 'application/json'
        }

    def get_api_host(self, api_source: str = 'modelhub') -> str:
        if api_source == 'grsai':
            return 'grsai.dakka.com.cn'
        else:
            return 'api.easyart.cc'

    def get_model_name(self, api_source: str = 'modelhub', model: str = 'nano banana pro') -> str:
        model_dict = {
            "grsai": {
                'nano banana 2': 'nano-banana-2',
                'nano banana pro': 'nano-banana-pro',
                'nano banana pro vt': 'nano-banana-pro-vt',
                'nano banana': 'nano-banana-fast',
            },
            "modelhub": {
                'nano banana 2': 'gemini-3.1-flash-image-preview',
                'nano banana pro': 'gemini-3-pro-image-preview',
                'nano banana pro vt': 'gemini-3-pro-image-preview',
                'nano banana': 'gemini-2.5-flash-image',
            }
        }
        model = model.lower()
        return model_dict[api_source][model]

    def main(self, api_source: str, model: str, mimeType: str, imageBase64: str, imageBase64_1: str, prompt: str, aspectRatio: str = '9:16', imageSize: str = '2k'):
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
        max_retries = 3
        retryable_statuses = {429, 502, 503, 504}
        last_exception = None
        data = b""
        res_status = None

        for attempt in range(max_retries):
            try:
                print("第一次尝试请求：", attempt == 0)
                if attempt == 0:
                    api_source = 'grsai'
                elif model == 'nano banana pro vt':
                    api_source = 'grsai'
                elif model == 'nano banana pro vt':
                    api_source = 'grsai'
                else:
                    api_source = 'modelhub'
                api_host = self.get_api_host(api_source)
                headers = self.get_header(api_source)
                model_name = self.get_model_name(api_source, model)
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
        print(data)
        if res_status != 200:
            raise RuntimeError(f"请求失败，重试后仍未成功: {last_exception}")

        data = json.loads(decoded_data)

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
                print("没有在 JSON 里找到 data:image/...;base64 的图片数据")
                return ("", "",)
            text = f"data:image/png;base64,{cleaned}"
            matches = list(pattern.finditer(text))
            if not matches:
                print("没有在 JSON 里找到 data:image/...;base64 的图片数据")
                return ("", "",)

        text = re.sub(r"data:image/[^;]+;base64,", "", text).strip()
        return (response, text,)
