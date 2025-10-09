from .core.minio_prodogape import MinioHandler

import os
import time
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
from openai import OpenAI
import requests
import json
import time


# minio_config = "minio_config.json"

def is_cn(text: str) -> bool:
    import re

    # 检测中文字符的正则表达式
    chinese_pattern = r'[\u4e00-\u9fff]'
    # 检测英文字符的正则表达式
    english_pattern = r'[a-zA-Z]'

    # 统计中文字符数量
    chinese_chars = len(re.findall(chinese_pattern, text))
    # 统计英文字符数量
    english_chars = len(re.findall(english_pattern, text))

    # 总字符数（只计算中英文字符）
    total_chars = chinese_chars + english_chars

    # 如果没有中英文字符，返回False
    if total_chars == 0:
        return False

    # 计算中文字符比例
    chinese_ratio = chinese_chars / total_chars

    # 如果中文字符比例大于50%，认为是中文
    # 如果中文字符比例等于0，认为是英文
    # 如果是混合文本，根据比例判断（这里设置阈值为0.5）
    return chinese_ratio > 0.5


def Load_minio_config():
    config_data = {
        "MINIO_HOST": os.environ.get("MINIO_HOST"),
        "MINIO_PORT": os.environ.get("MINIO_PORT"),
        "MINIO_ENDPOINT": os.environ.get("MINIO_ENDPOINT"),
        "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY"),
        "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY"),
        "COMFYINPUT_BUCKET": os.environ.get("COMFYINPUT_BUCKET"),
        "COMFYOUTPUT_BUCKET": os.environ.get("COMFYOUTPUT_BUCKET"),
        "MINIO_SECURE": os.environ.get("MINIO_SECURE"),
    }
    return config_data


class LoadImageFromMinio:

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        config_data = Load_minio_config()
        if config_data is not None:
            COMFYINPUT_BUCKET = os.environ.get("COMFYINPUT_BUCKET")
            minio_client = MinioHandler()
            if minio_client.is_minio_connected(COMFYINPUT_BUCKET):
                files = minio_client.get_all_files_in_bucket(COMFYINPUT_BUCKET)
        return {
            "required": {
                "image": (sorted(files),),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, image):
        config_data = Load_minio_config()
        if config_data is not None:
            minio_client = MinioHandler()
            if minio_client.is_minio_connected(config_data["COMFYINPUT_BUCKET"]):
                start_time = time.time()
                image_file = minio_client.get_file_by_name(
                    config_data["COMFYINPUT_BUCKET"], image
                )
                print(f"Minio get file time: {time.time()-start_time}s")

                i = Image.open(image_file)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if "A" in i.getbands():
                    mask = np.array(i.getchannel("A")).astype(
                        np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(mask)
                else:
                    mask = torch.zeros(
                        (64, 64), dtype=torch.float32, device="cpu")
                return (image, mask)
            else:
                raise Exception("Failed to connect to Minio")
        else:
            raise Exception(
                "Please check if your Minio is configured correctly")


class SaveImageToMinio:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "type": (
                    ["input", "output"],
                    {"default": "output"},
                ),
                "username": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "taskId": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("JSON",)

    def main(self, images, type, username, taskId, filename):
        results = []
        if username == "-1" or taskId == "-1" or filename == "-1":
            results.append({
                "success": False,
            })
            return results
        config_data = Load_minio_config()
        if config_data is not None:
            minio_client = MinioHandler()
            if (type == 'input'):
                bucket_name = config_data["COMFYINPUT_BUCKET"]
            if (type == 'output'):
                bucket_name = config_data["COMFYOUTPUT_BUCKET"]

            if minio_client.is_minio_connected(bucket_name):
                for image in images:
                    # file_name = f"{filename_prefix}-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid1()}.png"
                    file_name = f"{username}/{taskId}/{filename}.png"
                    i = 255. * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    buffer = BytesIO()
                    img.save(buffer, "png")
                    minio_client.put_image_by_stream(
                        bucket_name=bucket_name,
                        file_name=file_name,
                        file_stream=buffer,
                    )
                results.append({
                    "success": True,
                })
                return results
            else:
                raise Exception("Failed to connect to Minio")
        else:
            raise Exception(
                "Please check if your Minio is configured correctly")


class IsTextZhCN:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("BOOLEAN",)

    def main(self, text):
        return (is_cn(text),)


class OpenAIAPI:

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
                "key": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "host": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "model": (
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

    def main(self, data, key, host, model):

        client = OpenAI(api_key=key, base_url=host)
        import json
        messages = json.loads(data)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )

        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)


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


class UploadImageToNocodb:

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
        api_url = os.getenv("NOCODB_BASE_URL")
        xc_token = os.getenv("NOCODB_XC_TOKEN")
        # 准备请求头
        headers = {
            'xc-token': xc_token
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
        retry = 3
        while retry > 0:
            response = requests.post(
                f'{api_url}/api/v2/storage/upload', files=files, headers=headers, verify=False)
            retry -= 1
            if response.status_code != 200:
                print(f"上传失败，正在重试...({retry})")
                time.sleep(1)
                continue
            jsonData = response.json()
            print(jsonData)
            import json
            return (json.dumps(jsonData),)
        
        raise Exception('workflows run max retries exceeded')
