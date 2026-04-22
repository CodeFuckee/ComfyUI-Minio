import os
import json
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from .shared import is_cn


class DifyCn2En:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)

    def main(self, data):
        if not is_cn(data):
            return (data,)

        try:
            api_url = os.getenv("DIFY_API_URL")
            # 准备请求头
            headers = {
                'Content-Type': 'application/json'
            }

            # 准备请求体
            try:
                # 尝试解析inputs为JSON对象
                inputs_json = {
                    'text': data
                }
            except json.JSONDecodeError:
                return (data,)

            payload = {
                "inputs": inputs_json,
                "response_mode": 'blocking',
                "user": 'comfyui'
            }

            # 发送POST请求
            retry = 3
            while retry > 0:
                response = requests.post(
                    f'{api_url}/cn2en/v1/workflows/run', headers=headers, json=payload, verify=False)
                retry -= 1
                # 检查响应状态
                if response.status_code == 200:
                    return (response.json()['data']['outputs']['text'],)
                else:
                    error_message = f"请求失败，状态码: {response.status_code}, 响应: {response.text}"
                    print(error_message)
                    time.sleep(3)

            return (data,)

        except Exception as e:
            error_message = f"发送请求时出错: {str(e)}"
            print(error_message)
            return (data,)


class DifyImageDescribe:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)

    def main(self, images):
        api_url = os.getenv("DIFY_API_URL")
        # 准备请求头
        headers = {
            'Content-Type': 'application/json'
        }

        image = images[0]
        file_name = f"temp.png"
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, "png")
        # with open(file_name,'rb') as file:
        # 计算图片大小
        buffer_size = len(buffer.getvalue()) / (1024 * 1024)  # 转换为MB
        print(f"图片大小: {buffer_size:.2f}MB")
        files = {
            'file': (file_name, buffer.getvalue(), 'image/png'),
        }

        # 发送POST请求
        response = requests.post(
            f'{api_url}/describe2cn/v1/files/upload', files=files, data={
                'user': 'comfyui'
            }, verify=False)
        json = response.json()
        print(json)
        imageId = str(json['id'])

        payload = {
            "inputs": {},
            "response_mode": "blocking",
            "user": "comfyui",
            "files": [
                {
                    "transfer_method": "local_file",
                    "upload_file_id": imageId,
                    "type": "image",
                },
            ]
        }

        # 发送POST请求
        retry = 3
        while retry > 0:
            response = requests.post(
                f'{api_url}/describe2cn/v1/workflows/run', headers=headers, json=payload, verify=False)
            retry -= 1
            if response.status_code != 200:
                time.sleep(1)
                continue
            json = response.json()
            print(json)
            return (json['data']['outputs']['text'],)

        raise Exception('workflows run max retries exceeded')


class DifyImageDescribeEn:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)

    def main(self, images):
        api_url = os.getenv("DIFY_API_URL")
        # 准备请求头
        headers = {
            'Content-Type': 'application/json'
        }

        image = images[0]
        file_name = f"temp.png"
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, "png")
        # with open(file_name,'rb') as file:
        # 计算图片大小
        buffer_size = len(buffer.getvalue()) / (1024 * 1024)  # 转换为MB
        print(f"图片大小: {buffer_size:.2f}MB")
        files = {
            'file': (file_name, buffer.getvalue(), 'image/png'),
        }

        # 发送POST请求
        response = requests.post(
            f'{api_url}/describe2en/v1/files/upload', files=files, data={
                'user': 'comfyui'
            }, verify=False)
        json = response.json()
        print(json)
        imageId = str(json['id'])

        payload = {
            "inputs": {},
            "response_mode": "blocking",
            "user": "comfyui",
            "files": [
                {
                    "transfer_method": "local_file",
                    "upload_file_id": imageId,
                    "type": "image",
                },
            ]
        }

        # 发送POST请求
        retry = 3
        while retry > 0:
            response = requests.post(
                f'{api_url}/describe2en/v1/workflows/run', headers=headers, json=payload, verify=False)
            retry -= 1
            if response.status_code != 200:
                print(f'Error: {response.status_code}')
                print(response.text)
                time.sleep(3)
                continue
            json = response.json()
            print(json)
            return (json['data']['outputs']['text'],)

        raise Exception('workflows run max retries exceeded')
