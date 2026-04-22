import os
import re
import json
import time
import base64
import mimetypes
import http.client
import requests
from datetime import datetime
from .shared import download, generate_random_string


class VideoCombine:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line": (["fast", "stable"],),
                "model": (["veo-3.1-fast-generate-preview", "veo-3.1-generate-preview", "veo3.1", "veo3.1-4k", "veo3.1-fast", "veo3.1-pro", "veo3.1-pro-4k", "doubao-seedance-2-0-fast-260128", "doubao-seedance-2-0-260128"],),
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
                "second": (
                    "STRING",
                    {
                        "default": "5",
                    },
                ),
                "first_frame_url": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "end_frame_url": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "images_url": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "videos_url": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "audios_url": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "aspectRatio": (["1:1", "3:2", "16:9", "9:16", "4:3", "3:4", "21:9", "adaptive"],),
                "resolution": (["480p", "720p"],),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"

    RETURN_TYPES = ("STRING", 'STRING',)
    RETURN_NAMES = ("response", "image",)

    def get_api_host(self, line: str = 'stable') -> str:
        return 'api.easyart.cc'

    def base64_to_image(self, base64_str: str, save_path: str):
        """
        将 Base64 字符串解码为图片并保存到本地

        Args:
            base64_str: 图片的 base64 字符串（可带 data:image 前缀）
            save_path: 保存路径，如 "./output.png" 或 "test.jpg"
        """
        # 去掉 base64 前缀（data:image/png;base64, 这类）
        base64_data = re.sub(r'^data:image/\w+;base64,', '', base64_str)

        # 解码
        img_data = base64.b64decode(base64_data)

        # 写入文件
        with open(save_path, 'wb') as f:
            f.write(img_data)

        print(f"图片已保存至：{save_path}")

    def create_video_task_no_img(self, model: str = 'veo3.1', prompt: str = '', second: str = "8", aspectRatio: str = "16x9"):
        import http.client
        from codecs import encode

        if model in ['veo3.1', 'veo3.1-pro', 'veo3.1-fast', 'veo-3.1-fast-generate-preview', 'veo-3.1-generate-preview']:
            arrs = aspectRatio.split(':')
            if len(arrs) == 2:
                aspectRatio = str(arrs[0]) + "x" + str(arrs[1])

        conn = http.client.HTTPSConnection("api.easyart.cc")
        dataList = []
        boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
        dataList.append(encode('--' + boundary))
        dataList.append(encode('Content-Disposition: form-data; name=model;'))

        dataList.append(encode('Content-Type: {}'.format('text/plain')))
        dataList.append(encode(''))

        dataList.append(encode(model))
        dataList.append(encode('--' + boundary))
        dataList.append(encode('Content-Disposition: form-data; name=prompt;'))

        dataList.append(encode('Content-Type: {}'.format('text/plain')))
        dataList.append(encode(''))

        dataList.append(encode(prompt))
        dataList.append(encode('--' + boundary))
        dataList.append(encode('Content-Disposition: form-data; name=seconds;'))

        dataList.append(encode('Content-Type: {}'.format('text/plain')))
        dataList.append(encode(''))

        dataList.append(encode(second))
        dataList.append(encode('--' + boundary))
        dataList.append(encode('Content-Disposition: form-data; name=size;'))

        dataList.append(encode('Content-Type: {}'.format('text/plain')))
        dataList.append(encode(''))

        dataList.append(encode(aspectRatio))
        dataList.append(encode('--' + boundary))
        dataList.append(encode('Content-Disposition: form-data; name=watermark;'))

        dataList.append(encode('Content-Type: {}'.format('text/plain')))
        dataList.append(encode(''))

        dataList.append(encode("false"))
        dataList.append(encode('--' + boundary + '--'))
        dataList.append(encode(''))
        body = b'\r\n'.join(dataList)
        payload = body
        veo_key = os.getenv("EASYART_DEFAULT_API_KEY")
        headers = {
            'Authorization': f'Bearer {veo_key}',
            'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
        }
        print("正在提交任务...")
        conn.request("POST", "/v1/videos", payload, headers)
        res = conn.getresponse()
        data = res.read()
        json_text = data.decode("utf-8")
        print("响应:", json_text)
        return json.loads(json_text)

    def create_veo_task_with_img(self, model: str = 'veo3.1', prompt: str = '', second: str = "8", aspectRatio: str = "16x9", image_paths: list[str] = []):
        url = "https://api.easyart.cc/v1/videos"
        # 1. 准备普通的表单数据
        if model in ['veo3.1', 'veo3.1-pro', 'veo3.1-fast', 'veo-3.1-fast-generate-preview', 'veo-3.1-generate-preview']:
            arrs = aspectRatio.split(':')
            if len(arrs) == 2:
                aspectRatio = str(arrs[0]) + "x" + str(arrs[1])
        payload = {
            'model': model,
            'prompt': prompt,
            'seconds': second,
            'size': aspectRatio,
            'watermark': 'false'
        }

        files = []
        opened_files = []

        try:
            for i, path in enumerate(image_paths):
                if not os.path.exists(path):
                    print(f"警告: 找不到图片 {path}")
                    continue

                mime_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'
                f = open(path, 'rb')
                opened_files.append(f)

                field_name = 'input_reference'
                files.append((field_name, (os.path.basename(path), f, mime_type)))
            veo_key = os.getenv("EASYART_DEFAULT_API_KEY")
            headers = {
                'Authorization': f'Bearer {veo_key}'
            }

            print("正在提交任务...")
            response = requests.post(url, headers=headers, data=payload, files=files)
            print("响应:", response.text)

            return response.json()

        finally:
            for f in opened_files:
                f.close()

    def create_veo_task(self, model: str = 'veo3.1', prompt: str = '', second: str = "8", aspectRatio: str = "16x9", image_paths: list[str] = []):
        if len(image_paths) == 0:
            return self.create_video_task_no_img(model, prompt, second, aspectRatio)
        return self.create_veo_task_with_img(model, prompt, second, aspectRatio, image_paths)

    def poll_seedance_task_status(self, task_id) -> tuple[str, str]:
        url = f"https://api.easyart.cc/v1/videos/{task_id}"
        veo_key = os.getenv("EASYART_DEFAULT_API_KEY")
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {veo_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": "string"
        }

        while True:
            response = requests.get(url, headers=headers, json=payload)
            print(response.text)
            json_data = json.loads(response.text)
            if 'status' in json_data:
                if json_data['status'] == 'completed':
                    video_url = json_data['metadata']["url"]
                    return response.text, video_url
                elif json_data['status'] == 'error':
                    break
            elif 'error' in json_data:
                raise Exception(json_data['error']['code'])
            time.sleep(30)

        return "", ""

    def poll_veo_task_status(self, task_id) -> tuple[str, str]:
        url = f"https://api.easyart.cc/v1/videos/{task_id}"
        veo_key = os.getenv("EASYART_DEFAULT_API_KEY")
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {veo_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": "string"
        }

        while True:
            response = requests.get(url, headers=headers, json=payload)
            print(response.text)
            json_data = json.loads(response.text)
            if 'status' in json_data:
                if json_data['status'] == 'completed':
                    video_url = json_data['video_url']
                    return response.text, video_url
                elif json_data['status'] == 'error':
                    break
            elif 'error' in json_data:
                raise Exception(json_data['error']['code'])
            time.sleep(30)

        return "", ""

    def create_seedance_task(self, model: str = 'doubao-seedance-2-0-fast-260128', prompt: str = '', second: int = 5, ratio: str = 'adaptive', first_frame_url: str = '', end_frame_url: str = '', images_url: list = [], videos_url: list = [], audios_url: list = []):
        seedance_key = os.getenv("EASYART_DEFAULT_API_KEY")
        conn = http.client.HTTPSConnection("api.easyart.cc")
        content = [
            {
                "type": "text",
                "text": prompt
            },
        ]
        if first_frame_url != "":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": first_frame_url
                    },
                    "role": "first_frame"
                }
            )
        if end_frame_url != "":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": end_frame_url
                    },
                    "role": "last_frame"
                }
            )
        for image in images_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    },
                    "role": "reference_image"
                }
            )
        for video in videos_url:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": video
                    },
                    "role": "reference_video"
                }
            )
        for audio in audios_url:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": audio
                    },
                    "role": "reference_audio"
                }
            )
        payload = json.dumps({
            "model": model,
            "content": content,
            "generate_audio": True,
            "ratio": ratio,
            "duration": int(second),
            "watermark": False
        })
        headers = {
            'Authorization': f'Bearer {seedance_key}',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/videos", payload, headers)
        res = conn.getresponse()
        data = res.read()
        text = data.decode("utf-8")
        print(text)
        json_data = json.loads(text)
        return json_data

    def get_time_random_str(self) -> str:
        """
        生成字符串：当前时间（精确到毫秒） + 16位随机数字
        格式示例：20250320145832123_1234567890123456
        """
        now = datetime.now()
        time_str = now.strftime("%Y%m%d%H%M%S%f")[:-3]
        random_16 = generate_random_string(16)
        result = f"{time_str}_{random_16}"
        return result

    def str2urls(self, text: str) -> list[str]:
        result = []
        if text == "":
            result = []
        else:
            try:
                result = json.loads(text)
            except:
                result = []
        return result

    def main(self, line: str, model: str, imageBase64: str, imageBase64_1: str, prompt: str, second: str, aspectRatio: str = '9:16', resolution: str = '480p', first_frame_url='', end_frame_url='', images_url="", videos_url="", audios_url=""):
        image_paths = []
        if imageBase64 != '-1' and imageBase64 != '':
            file_name = "temp/" + self.get_time_random_str() + ".png"
            self.base64_to_image(imageBase64, file_name)
            image_paths.append(file_name)

        if imageBase64_1 != '-1' and imageBase64_1 != '':
            file_name = "temp/" + self.get_time_random_str() + ".png"
            self.base64_to_image(imageBase64, file_name)
            image_paths.append(file_name)

        max_retries = 1
        task_id = ""
        result = ""
        for attempt in range(max_retries):
            try:
                if 'veo' in model:
                    result = self.create_veo_task(model, prompt, second, aspectRatio, image_paths)
                elif 'seedance' in model:
                    images_url = self.str2urls(images_url)
                    videos_url = self.str2urls(videos_url)
                    audios_url = self.str2urls(audios_url)
                    result = self.create_seedance_task(model, prompt, second, aspectRatio, first_frame_url, end_frame_url, images_url, videos_url, audios_url)
                if 'id' in result:
                    task_id = result['id']
                    break
                elif 'error' in result:
                    raise Exception(result['error']['message'])
                else:
                    print("任务提交失败，未能获取到任务ID")

            except Exception as e:
                print('请求失败' + str(e))
                if attempt < (max_retries - 1):
                    time.sleep(2 ** attempt)
                    continue
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        print(f"任务提交成功，任务ID: {task_id}")
        print("开始轮询任务状态...")
        if 'seedance' in model:
            response_text, video_url = self.poll_seedance_task_status(task_id)
        else:
            response_text, video_url = self.poll_veo_task_status(task_id)
        if video_url == "":
            with open("debug/error_" + self.get_time_random_str() + ".json", "w+") as f:
                f.write(response_text)
            raise Exception("生成失败")

        video_path = os.path.abspath(os.path.join("temp", self.get_time_random_str() + ".mp4"))
        download(video_url, video_path)
        return json.dumps(result), video_path
