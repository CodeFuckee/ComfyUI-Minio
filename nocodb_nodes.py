import os
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO


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
