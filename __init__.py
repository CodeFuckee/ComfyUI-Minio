from .node import *

NODE_CLASS_MAPPINGS = {
    # "Set Minio Config": SetMinioConfig,
    "Load Image From Minio": LoadImageFromMinio,
    "Save Image To Minio": SaveImageToMinio,
    "Is Text Is CN": IsTextZhCN,
    "OpenAI API": OpenAIAPI,
    "Dify Cn TO En": DifyCn2En,
    "Dify Image Describe": DifyImageDescribe,
    "Dify Image Describe En": DifyImageDescribeEn,
    "Upload Image To Nocodb": UploadImageToNocodb,
    "Sam Image Predict": SamImagePredict,
    "Nano Banana Pro": NanoBananaPro,
    "Nano Banana Pro 2 Pic": NanoBananaPro2,
    "Nano Banana Pro Combine": NanoBananaProCombine,
    "Nano Banana Pro Combine V2": NanoBananaProCombine2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # "Set Minio Config": "Set Minio Config",
    "Load Image From Minio": "Load Image From Minio",
    "Save Image To Minio": "Save Image To Minio",
    "Is Text Is CN": "Is Text Is CN",
    "OpenAI API": "OpenAI API",
    "Dify Cn TO En": "Dify Cn TO En",
    "Dify Image Describe": "Dify Image Describe",
    "Dify Image Describe En": "Dify Image Describe En",
    "Upload Image To Nocodb": "Upload Image To Nocodb",
    "Sam Image Predict": "Sam Image Predict",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
