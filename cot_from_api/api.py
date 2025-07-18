'''
工作计划：
1、调API
2、找prompt
3、先用单图片进行CoT调试
4、数据预处理做好,看看能不能将其预处理成一个文件
'''
# ② 在代码里读取环境变量，然后发请求
import os
import requests
import json
from openai import OpenAI
import os
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-5MvaVcI62IFzE0gbFDaDYeOksJsoxKOHcHs5JC9NFkuTGRSZ"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
client = OpenAI(
    # 下面两个参数的默认值来自环境变量，可以不加
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

import base64

with open("./1752660927641.jpg", "rb") as image_file:
    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "针对这张图片和提问，请输出推理时需要关注的关键点提示。"
                        "要求："
                        "1. 用简洁的短句描述关键点；"
                        "2. 按照1、2、3编号输出；"
                        "3. 不要解释，不要展开，不输出答案。"
                        "提问：这张图片中谁处于危险？"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    },
                },
            ]
        }
    ],
    max_tokens=500,
)

print(response.choices[0])
