import os
import sys
import json
import base64
import mimetypes
import http.client
import urllib.parse


# def _str2bool(v):
#     if isinstance(v, bool):
#         return v
#     s = str(v).strip().lower()
#     return s in ("1", "true", "yes", "y", "on")


# def _post_json(url, headers, payload):
#     u = urllib.parse.urlparse(url)
#     host = u.netloc
#     path = u.path or "/"
#     if u.query:
#         path = f"{path}?{u.query}"
#     conn = http.client.HTTPSConnection(host, timeout=120)
#     body = json.dumps(payload)
#     conn.request("POST", path, body=body, headers=headers)
#     resp = conn.getresponse()
#     data = resp.read()
#     conn.close()
#     return resp.status, data


# def main():
#     api_url = os.environ.get("GRSAI_API_URL", "https://api.easyart.cc/v1/videos")
#     api_key = os.environ.get("GRSAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "sk-mCf8eomJZbcZga4ZBJPRx0ceb4s1888EBa55Qd7nOW451QXD")
#     model = os.environ.get("OPENAI_VIDEO_MODEL", "sora-2")
#     prompt = os.environ.get("OPENAI_VIDEO_PROMPT", "让这张图动起来")
#     seconds_env = os.environ.get("OPENAI_VIDEO_DURATION", "10")
#     size = os.environ.get("OPENAI_VIDEO_SIZE", "720x1280")
#     watermark = _str2bool(os.environ.get("OPENAI_VIDEO_WATERMARK", "false"))
#     private = _str2bool(os.environ.get("OPENAI_VIDEO_PRIVATE", "false"))
#     image_path = os.environ.get("OPENAI_VIDEO_IMAGE", "image_1774497357708.png")
#     if not api_key:
#         print("请设置 GRSAI_API_KEY 或 OPENAI_API_KEY", file=sys.stderr)
#         sys.exit(2)
#     try:
#         seconds = float(seconds_env)
#     except Exception:
#         seconds = 10.0
#     if not os.path.exists(image_path):
#         print(f"找不到图片文件: {image_path}", file=sys.stderr)
#         sys.exit(2)
#     mime = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
#     with open(image_path, "rb") as f:
#         img_b64 = base64.b64encode(f.read()).decode("utf-8")
#     data_url = f"data:{mime};base64,{img_b64}"
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "seconds": str(seconds),
#         # "url": data_url,
#         "input_reference": 'image_1774497357708.png',
#         "size": size,
#         "watermark": watermark,
#         # "private": private,
#     }
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }
#     status, data = _post_json(api_url, headers, payload)
#     text = data.decode("utf-8", errors="ignore")
#     print(text)
#     if status >= 400:
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

import os
import sys
import json
import base64
import mimetypes
import http.client
import urllib.parse


def _get_with_body(url, headers, payload):
    u = urllib.parse.urlparse(url)
    host = u.netloc
    path = u.path or "/"
    if u.query:
        path = f"{path}?{u.query}"
    body = json.dumps(payload)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(host, timeout=60)
    else:
        conn = http.client.HTTPConnection(host, timeout=60)
    conn.request("GET", path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    status = resp.status
    conn.close()
    return status, data


def main():
    url = os.environ.get(
        "EASYART_TASK_URL",
        "https://api.easyart.cc/v1/videos/sora-2:task_aI012FSVyJOye5Zfw1N1fBZsPN0U1Chr",
    )
    token = os.environ.get("EASYART_API_KEY", 'sk-mCf8eomJZbcZga4ZBJPRx0ceb4s1888EBa55Qd7nOW451QXD') or os.environ.get("GRSAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not token:
        print("缺少令牌，请设置 EASYART_API_KEY 或 GRSAI_API_KEY 或 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"model": "sora-2"}
    status, data = _get_with_body(url, headers, payload)
    text = data.decode("utf-8", errors="ignore")
    print(text)
    if status >= 400:
        sys.exit(1)


if __name__ == "__main__":
    main()

