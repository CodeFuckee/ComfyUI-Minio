import os
import re
import json
import time
from turtle import mode
import torch
import random
import base64
import string
import secrets
import requests
import mimetypes
import http.client
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from .core.minio_prodogape import MinioHandler
from transformers import Sam2Processor, Sam2Model

if not os.path.exists("debug"):
    os.mkdir("debug")
if not os.path.exists("temp"):
    os.mkdir("temp")

def download(img_url: str, save_path: str) -> bool:
    count = 0
    while count <= 5:
        try:
            response = requests.get(img_url, stream=True)
            if response.status_code == 200:
                if os.path.exists(save_path):
                    return True
                print(save_path)
                with open(save_path, "wb") as f:
                    f.write(response.content)
            time.sleep(3)
            return True
        except Exception as e:
            count += 1
            print(str(e))
            time.sleep(3)
    return False

def generate_random_string(length=128):
    """生成指定长度的随机字符串，默认128位"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


# minio_config = "minio_config.json"

def is_cn(text: str) -> bool:
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

def process_sam2_segmentation(image, points, model, processor, device):
    """使用SAM2模型处理分割"""
    if not points:
        return None
    
    # 转换点为所需格式: [image_dim, object_dim, point_per_object_dim, coordinates]
    input_points = [[points]]  # 单个对象多个点
    input_labels = [[[1] * len(points)]]  # 所有点都是正样本
    
    inputs = processor(images=image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    return masks

def save_results(image_array, masks, points, output_dir, output_type="all", mask_indices=None):
    """保存分割结果
    
    Args:
        image_array: 原始图像数组
        masks: 分割掩码
        points: 输入点坐标
        output_dir: 输出目录
        output_type: 输出类型，可选值为 "all", "combined", "individual", "none"
        mask_indices: 指定要输出的掩码序号列表，如果为None则输出所有掩码
    """
    if output_type == "none":
        print("根据设置，不输出任何图像")
        return
        
    num_masks = min(3, masks.shape[1])
    saved_combined = False
    saved_individual = False
    
    # 保存组合图
    if output_type in ["all", "combined"]:
        # 计算图像尺寸
        height, width = image_array.shape[:2]
        combined_width = width * (num_masks + 1)
        combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
        
        # 原始图像和点
        original_with_points = image_array.copy()
        # if points:
        #     points_array = np.array(points)
        #     # 在图像上绘制点标记
        #     for i, (x, y) in enumerate(points):
        #         # 绘制红色圆点
        #         cv2.circle(original_with_points, (int(x), int(y)), 5, (255, 0, 0), -1)
        
        # combined_image[:, :width] = original_with_points
        
        # 掩码
        for i in range(num_masks):
            mask = masks[0, i].numpy()
            overlay = image_array.copy()
            overlay[mask > 0.5] = [255, 0, 0]  # 红色标记掩码
            blended = (0.7 * image_array + 0.3 * overlay).astype(np.uint8)
            
            start_x = width * (i + 1)
            end_x = width * (i + 2)
            combined_image[:, start_x:end_x] = blended
        
        # 直接保存图像数组，不使用matplotlib
        Image.fromarray(combined_image).save(f"{output_dir}/sam2_segmentation_results.png")
        saved_combined = True
    
    # 单独保存每个掩码
    if output_type in ["all", "individual"]:
        # 确定要保存的掩码索引
        indices_to_save = range(num_masks)
        if mask_indices is not None:
            # 过滤有效的掩码索引
            indices_to_save = [idx for idx in mask_indices if 0 <= idx < num_masks]
            if not indices_to_save:
                print(f"警告: 指定的掩码序号 {mask_indices} 超出范围 [0-{num_masks-1}]，将不保存任何单独掩码")
        
        # 保存指定的掩码
        for i in indices_to_save:
            mask = masks[0, i].numpy()
            mask_overlay = image_array.copy()
            mask_overlay[mask > 0.5] = [255, 0, 0]
            blended = (0.7 * image_array + 0.3 * mask_overlay).astype(np.uint8)
            
            # 在图像上绘制点标记（如果有的话）
            # if points:
            #     points_array = np.array(points)
            #     for j, (x, y) in enumerate(points):
            #         # 绘制黄色圆点
            #         cv2.circle(blended, (int(x), int(y)), 5, (0, 255, 255), -1)
            
            # 直接保存图像数组，不使用matplotlib
            Image.fromarray(blended).save(f"{output_dir}/mask_{i+1}.png")
            
        
        saved_individual = len(indices_to_save) > 0
    
    # 输出保存信息
    if saved_combined and saved_individual:
        if mask_indices:
            print(f"已保存组合结果图和指定的掩码图")
        else:
            print(f"已保存组合结果图和 {num_masks} 个单独掩码图")
    elif saved_combined:
        print(f"已保存组合结果图")
    elif saved_individual:
        if mask_indices:
            print(f"已保存指定的掩码图")
        else:
            print(f"已保存 {num_masks} 个单独掩码图")
    else:
        print("未保存任何图像")

def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


class SamImagePredict:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "indics": ("INT",{"default": 1, "min": 0, "max": 2, "step": 1}),
                "x": ("INT",{"default": 200, "min": 0, "step": 1}),
                "y": ("INT",{"default": 300, "min": 0, "step": 1}),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"

    RETURN_TYPES = ("IMAGE", 'MASK', "STRING",)
    RETURN_NAMES = ("images", "masks", "files",)

    def main(self, images: torch.Tensor, indics: int = 1, x: int = 200, y: int = 300):
        # 初始化模型
        print("正在加载SAM2模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 设置模型下载路径
        model_url = "https://hf-mirror.com/spaces/fffiloni/SAM2-Image-Predictor/resolve/main/checkpoints/sam2_hiera_large.pt"
        model_dir = Path("./checkpoints")
        model_path = model_dir / "sam2_hiera_large.pt"
    
        # 创建模型目录
        model_dir.mkdir(exist_ok=True)
    
        # 下载模型（如果不存在）
        if not model_path.exists():
            print(f"下载模型文件: {model_url}")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"模型已下载到: {model_path}")
        else:
            print(f"使用本地模型: {model_path}")
    
        # 加载模型和处理器
        # 直接使用本地模型文件
        model_state_dict = torch.load(model_path, map_location=device)
        model = Sam2Model.from_pretrained("facebook/sam2-hiera-large", trust_remote_code=True, ignore_mismatched_sizes=True).to(device)
        # 加载自定义权重
        model.load_state_dict(model_state_dict, strict=False)
        processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large", trust_remote_code=True)
        print(f"模型已加载，使用设备: {device}")
        # 类型转换：将 [B,H,W,C] 或 [H,W,C] 转为 PIL.Image
        if images.ndim == 4:
            tensor_image = images[0]
        else:
            tensor_image = images
        numpy_image = (tensor_image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        raw_image = Image.fromarray(numpy_image)
        image_array = np.array(raw_image)

        # 获取点坐标
        os.makedirs('results', exist_ok=True)
        
        points = []
        points.append([x, y])
        
        masks = process_sam2_segmentation(raw_image, points, model, processor, device)
    
        if masks is None or masks.shape[1] == 0:
            print("分割失败，未生成有效掩码")
            return
    
        print(f"生成了 {masks.shape[1]} 个掩码")
        
        # 解析掩码序号
        mask_indices: list[int] = [indics]
        
        # 保存结果
        save_results(image_array, masks, points, 'results', 'individual', mask_indices)
        indices_to_use = range(masks.shape[1])
        if mask_indices is not None:
            valid_indices = [idx for idx in mask_indices if 0 <= idx < masks.shape[1]]
            indices_to_use = valid_indices if len(valid_indices) > 0 else indices_to_use
        combined_mask = np.zeros(image_array.shape[:2], dtype=bool)
        for i in indices_to_use:
            m = masks[0, i].cpu().numpy()
            combined_mask |= (m > 0.5)
        overlay = image_array.copy()
        overlay[combined_mask] = [255, 0, 0]
        blended = (0.7 * image_array + 0.3 * overlay).astype(np.uint8)
        output_image = torch.from_numpy(blended.astype(np.float32) / 255.0)
        # 将combined_mask转换为与output_image维度、类型一致的变量
        combined_mask_rgb = np.stack([combined_mask] * 3, axis=-1).astype(np.float32)
        combined_mask_tensor = torch.from_numpy(combined_mask_rgb)
        if images.ndim == 4:
            output_image = output_image.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            combined_mask_tensor = combined_mask_tensor.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
        else:
            output_image = output_image.unsqueeze(0)
            combined_mask_tensor = combined_mask_tensor.unsqueeze(0)
        
        # return (output_image, '',)
        try:
            combined_mask_tensor = pil2mask(image_array)
        except Exception as e:
            print(str(e))

        try:
            combined_mask_tensor = pil2mask(blended)
        except Exception as e:
            print(str(e))

        try:
            combined_mask_tensor = pil2mask(Image.fromarray(combined_mask.astype(np.uint8)))
        except Exception as e:
            print(str(e))
        
        return (output_image , masks[0][indics], '',)


class NanoBananaPro:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def main(self, mimeType: str, imageBase64: str, prompt: str, aspectRatio: str = '9:16', imageSize: str = '2k'):
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
                                "mime_type": mimeType,
                                "data": imageBase64,
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

        # 取 candidates[0].content.parts[*].text，里面是 markdown
        parts = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
        )

        text = "\n".join(p.get("inlineData", {}).get('data','') for p in parts if isinstance(p, dict))
        response = "".join(p.get("text") for p in parts if isinstance(p, dict) and 'text' in p)
        # 匹配 data:image/<ext>;base64,<payload>
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
        return (response , text,)


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

        # 取 candidates[0].content.parts[*].text，里面是 markdown
        parts = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
        )

        text = "\n".join(p.get("inlineData", {}).get('data','') for p in parts if isinstance(p, dict))
        response = "".join(p.get("text") for p in parts if isinstance(p, dict) and 'text' in p)
        # 匹配 data:image/<ext>;base64,<payload>
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
        return (response , text,)


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
                print('请求失败'+str(e))
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
        # with open("response.json", "w", encoding="utf-8") as f:
        #     f.write(decoded_data)
        print(data)
        if res_status != 200:
            raise RuntimeError(f"请求失败，重试后仍未成功: {last_exception}")

        # JSON_PATH = Path("response.json")
        # OUT_DIR = Path(".")
        # OUT_DIR.mkdir(parents=True, exist_ok=True)

        data = json.loads(decoded_data)

        # 取 candidates[0].content.parts[*].text，里面是 markdown
        parts = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
        )

        text = "\n".join(p.get("inlineData", {}).get('data','') for p in parts if isinstance(p, dict))
        response = "".join(p.get("text") for p in parts if isinstance(p, dict) and 'text' in p)
        # 匹配 data:image/<ext>;base64,<payload>
        pattern = re.compile(r"data:image/(?P<ext>png|jpeg|jpg|webp|gif);base64,(?P<b64>[A-Za-z0-9+/=\s]+)")

        matches = list(pattern.finditer(text))
        if not matches:
            cleaned = re.sub(r"\s+", "", text)
            if not cleaned:
                print("没有在 JSON 里找到 data:image/...;base64 的图片数据")
                return ("", "", )
                # raise SystemExit("没有在 JSON 里找到可用的 base64 图片数据")
            text = f"data:image/png;base64,{cleaned}"
            matches = list(pattern.finditer(text))
            if not matches:
                print("没有在 JSON 里找到 data:image/...;base64 的图片数据")
                return ("", "", )
                # raise SystemExit("没有在 JSON 里找到 data:image/...;base64 的图片数据")

        # for i, m in enumerate(matches, start=1):
        #     # b64_payload = re.sub(r"\s+", "", m.group("b64"))
        #     # img_bytes = base64.b64decode(b64_payload)
        #     tt = generate_random_string()
        #     # out_path = OUT_DIR / f"extracted_image_{tt}.png"
        #     # out_path.write_bytes(img_bytes)
        #     # print(f"saved: {out_path.resolve()}")

        text = re.sub(r"data:image/[^;]+;base64,", "", text).strip()
        return (response , text,)

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
                print('请求失败'+str(e))
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
        # print(data)
        if res_status != 200:
            raise RuntimeError(f"请求失败，重试后仍未成功: {last_exception}")
        data = json.loads(decoded_data)
        parts = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
        )

        text = "\n".join(p.get("inlineData", {}).get('data','') for p in parts if isinstance(p, dict))
        response = "".join(p.get("text") for p in parts if isinstance(p, dict) and 'text' in p)
        # 匹配 data:image/<ext>;base64,<payload>
        pattern = re.compile(r"data:image/(?P<ext>png|jpeg|jpg|webp|gif);base64,(?P<b64>[A-Za-z0-9+/=\s]+)")

        matches = list(pattern.finditer(text))
        if not matches:
            cleaned = re.sub(r"\s+", "", text)
            if not cleaned:
                print("没有在 JSON 里找到 data:image/...;base64 的图片数据")
                return ("", "", )
            text = f"data:image/png;base64,{cleaned}"
            matches = list(pattern.finditer(text))
            if not matches:
                print("没有在 JSON 里找到 data:image/...;base64 的图片数据")
                return ("", "", )
        text = re.sub(r"data:image/[^;]+;base64,", "", text).strip() 
        try:
            try:
                img_bytes = base64.b64decode(text, validate=True)
            except Exception:
                img_bytes = base64.urlsafe_b64decode(text + "===")
            img = Image.open(BytesIO(img_bytes))
            img.load()
            buf = BytesIO()
            img.save(buf, format="PNG")
            png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return (response, png_b64,)
        except Exception:
            ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
            rand6 = f"{secrets.randbelow(1000000):06d}"
            debug_name = f"{ts}{rand6}.json"
            Path(debug_name).write_text(
                json.dumps({"response": response, "data": data}, ensure_ascii=False),
                encoding="utf-8",
            )
            return (response, text,)




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
        dataList.append(encode('--'+boundary+'--'))
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
    
    # 视频生成的比例
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
                
                # 默认使用 'input_reference' 作为字段名
                # 如果 API 要求多张图使用不同字段名（如 input_reference_1, input_reference_2），可修改为 f'input_reference_{i+1}'
                field_name = 'input_reference' 
                files.append((field_name, (os.path.basename(path), f, mime_type)))
            veo_key = os.getenv("EASYART_DEFAULT_API_KEY")
            headers = {
                'Authorization': f'Bearer {veo_key}'
                # 注意：使用 requests 时不要手动设置 Content-Type 为 multipart/form-data
                # requests 会自动设置正确的 Content-Type 并带上 boundary
            }

            print("正在提交任务...")
            response = requests.post(url, headers=headers, data=payload, files=files)
            print("响应:", response.text)
            
            return response.json()
            
        finally:
            # 确保文件被关闭
            for f in opened_files:
                f.close()
    
    def create_veo_task(self, model: str = 'veo3.1', prompt: str = '', second: str = "8", aspectRatio: str = "16x9", image_paths: list[str] = []):
        if len(image_paths) == 0:
            return self.create_video_task_no_img(model, prompt, second, aspectRatio)
        return self.create_veo_task_with_img(model, prompt, second, aspectRatio, image_paths)
    
    def poll_task_status(self, task_id) -> tuple[str, str]:
        url = f"https://api.easyart.cc/v1/videos/{task_id}"
        veo_key = os.getenv("EASYART_DEFAULT_API_KEY")
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {veo_key}',
            'Content-Type': 'application/json'
        }
        # 原代码中 GET 请求带了 body
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

    def create_seedance_task(self, model: str = 'doubao-seedance-2-0-fast-260128', prompt: str = '', second: int = 5, ratio: str = 'adaptive', first_frame_url: str = '', end_frame_url: str ='', images_url: list = [], videos_url: list = [], audios_url: list = []):
        seedance_key= os.getenv("EASYART_DEFAULT_API_KEY")
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
        # 1. 获取当前时间，精确到毫秒，格式：年月日时分秒毫秒
        now = datetime.now()
        time_str = now.strftime("%Y%m%d%H%M%S%f")[:-3]  # 截取到毫秒（3位）
        # 2. 生成 16 位随机数字字符串
        random_16 = generate_random_string(16)
        # 3. 拼接成最终字符串（用下划线分隔，方便阅读）
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

    def main(self, line: str, model: str, imageBase64: str, imageBase64_1: str, prompt: str, second: str, aspectRatio: str = '9:16', resolution: str = '480p', first_frame_url = '', end_frame_url = '', images_url = "", videos_url = "", audios_url = ""):
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
                print('请求失败'+str(e))
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
        response_text, video_url = self.poll_task_status(task_id)
        if video_url == "":
            with open("debug/error_" + self.get_time_random_str()+".json", "w+") as f:
                f.write(response_text)
            raise Exception("生成失败")
        
        video_path = os.path.abspath(os.path.join("temp", self.get_time_random_str() + ".mp4"))
        download(video_url,video_path)
        return json.dumps(result), video_path