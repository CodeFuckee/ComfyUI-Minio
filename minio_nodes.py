import os
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from .core.minio_prodogape import MinioHandler


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
