from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image
import numpy as np
import os
import requests
from pathlib import Path
from .core.minio_prodogape import MinioHandler

import time
import torch
from io import BytesIO
from openai import OpenAI
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