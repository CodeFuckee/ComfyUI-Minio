import os
import re
import time
import string
import secrets
import requests


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
