import os

if not os.path.exists("debug"):
    os.mkdir("debug")
if not os.path.exists("temp"):
    os.mkdir("temp")

from .minio_nodes import LoadImageFromMinio, SaveImageToMinio
from .text_nodes import IsTextZhCN
from .openai_nodes import OpenAIAPI
from .dify_nodes import DifyCn2En, DifyImageDescribe, DifyImageDescribeEn
from .nocodb_nodes import UploadImageToNocodb
from .sam_nodes import SamImagePredict
from .nanobanana_pro import NanoBananaPro
from .nanobanana_pro2 import NanoBananaPro2
from .nanobanana_pro_combine import NanoBananaProCombine
from .nanobanana_pro_combine2 import NanoBananaProCombine2
from .video_nodes import VideoCombine

__all__ = [
    "LoadImageFromMinio",
    "SaveImageToMinio",
    "IsTextZhCN",
    "OpenAIAPI",
    "DifyCn2En",
    "DifyImageDescribe",
    "DifyImageDescribeEn",
    "UploadImageToNocodb",
    "SamImagePredict",
    "NanoBananaPro",
    "NanoBananaPro2",
    "NanoBananaProCombine",
    "NanoBananaProCombine2",
    "VideoCombine",
]
