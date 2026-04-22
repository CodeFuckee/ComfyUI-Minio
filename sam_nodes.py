import os
import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import Sam2Processor, Sam2Model


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
                "indics": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
                "x": ("INT", {"default": 200, "min": 0, "step": 1}),
                "y": ("INT", {"default": 300, "min": 0, "step": 1}),
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

        return (output_image, masks[0][indics], '',)
