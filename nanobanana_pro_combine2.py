import os
import re
import json
import time
import base64
import secrets
import requests
import http.client
from pathlib import Path


class NanoBananaProCombine2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line": (["fast", "stable", "economy", "cheap"],),
                "model": (["nano banana 2", "nano banana pro", "gpt image 2"],),
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
                "images_url": (
                    "STRING",
                    {
                        "default": "",
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
    
    def save_response_when_except(self, decoded_data: str) -> str:
        ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
        rand6 = f"{secrets.randbelow(1000000):06d}"
        debug_name = f"{ts}{rand6}.json"
        Path(debug_name).write_text(
            decoded_data,
            encoding="utf-8",
        )
        return debug_name
    
    def handle_image_url_result(self, decoded_data: str, img_url: str):
        try:
            import urllib.request
            print(f'结果图片{img_url}')
            with urllib.request.urlopen(img_url) as resp:
                img_data = resp.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return (decoded_data, img_base64,)
        except Exception as e:
            json_path = self.save_response_when_except(decoded_data)
            print(str(e) + json_path)
            raise

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
                            return self.handle_image_url_result(decoded_data, img_url)
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
        self.save_response_when_except(decoded_data)
        raise ValueError(f"no img debug file {debug_name}")
    
    def handle_banana(self, line: str, model: str, mimeType: str, imageBase64: str, imageBase64_1: str, prompt: str, aspectRatio: str = '9:16', imageSize: str = '2k'):
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
                print(f'状态码{res_status} decoded_data_test:{decoded_data_test}')
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
    
    def handle_gpt_image(self, line: str, imageUrls: list[str], prompt: str, aspectRatio: str):
        decode_data = ''
        result_data = ''
        if line == 'cheap':
            api_key = os.getenv("GRSAI_API_KEY")
            if not api_key:
                raise RuntimeError("缺少环境变量 GRSAI_API_KEY")
            url = "https://grsai.dakka.com.cn/v1/draw/completions"
            payload = {
                "model": "gpt-image-2",
                "prompt": prompt,
                "n": 1,
                "response_format": "url",
                "image": imageUrls,
                "size": aspectRatio,
                "webHook": "https://douniwan.com"
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            # with open("response.json", "w", encoding="utf-8") as f:
            #     json.dump(data, f, ensure_ascii=False, indent=2)
            decode_data = json.dumps(data, ensure_ascii=False, indent=2)
            print(decode_data)
            deadline = time.time() + 600
            while True:
                if time.time() > deadline:
                    break
                if 'data' in data and 'id' in data['data']:
                    id = data['data']['id']
                    result_url = "https://grsai.dakka.com.cn/v1/draw/result"
                    result_response = requests.post(result_url, headers=headers, json={"id":id}, timeout=600)
                    result_data = result_response.json()
                    if 'data' in result_data and 'status' in result_data['data']:
                        status = result_data['data']['status']
                        if status == 'success' or status == 'succeeded':
                            img_results = result_data['data']['results']
                            for img_result in img_results:
                                if 'url' in img_result:
                                    url = img_result['url']
                                    return self.handle_image_url_result(decode_data, url)
                        elif status == 'failed':
                            raise TimeoutError(f"运行失败 任务提交响应数据:{decode_data} result:{result_data}")
                    time.sleep(10)
            raise TimeoutError(f"超时 任务提交响应数据:{decode_data} result:{result_data}")
        else:
            api_key = os.getenv("EASYART_API_KEY")
            if not api_key:
                raise RuntimeError("缺少环境变量 EASYART_API_KEY")
            url = "https://api.easyart.cc/v1/images/generations"
            payload = {
                "model": "gpt-image-2",
                "prompt": prompt,
                "n": 1,
                "response_format": "url",
                "image": imageUrls,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            if 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    img_url = item['url']
                    return self.handle_image_url_result(json.dumps(data), img_url)
            else:
                raise ValueError(f"响应数据没有图片 {data}")
        
        raise Exception(f"未知错误 {line} {imageUrls} {prompt} {aspectRatio}")

    def main(self, line: str, model: str, mimeType: str, imageBase64: str, imageBase64_1: str, images_url: str, prompt: str, aspectRatio: str = '9:16', imageSize: str = '2k'):
        if 'nano banana' in model:
            return self.handle_banana(line, model, mimeType, imageBase64, imageBase64_1, prompt, aspectRatio, imageSize)
        elif 'gpt image' in model:
            if images_url == '-1' or images_url == '':
                imageUrls = []
            else:
                imageUrls = json.loads(images_url)
            return self.handle_gpt_image(line, imageUrls, prompt, aspectRatio)

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
