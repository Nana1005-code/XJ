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

with open("/data/aovkqa/train2017/000000012991.jpg", "rb") as image_file:
    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "提问：图像中的人物的性别是什么。针对这个包含图像和提问的输入，请深入理解意义，think step by step，告诉我应该如何去推理，只输出推理的指引，不输出推理的具体细节，也不需要告诉我答案"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    },
                },
            ],
        }
    ],
    max_tokens=1000,
)

print(response.choices[0])

'''
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "提问：在图片中的展览馆的下一届世博会开展地是哪里？针对这个包含图像和文本的输入，请深入理解意义，think step by step，告诉我应该如何去推理，只输出推理的指引，不输出推理的具体细节，也不需要告诉我答案"},#在图片中的展览馆的下一届世博会开展地是哪里?请根据文本提问和图片,生成一个推理链条CoT。
                {
                    "type": "image_url",
                    "image_url": "https://th.bing.com/th/id/OIP.pTU-7xHh8y_HmHihLlI4vwHaFf?r=0&o=7rm=3&rs=1&pid=ImgDetMain",
                },
            ],
        }
    ],
    max_tokens=1000,
)

print(response.choices[0])


completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "面向图文问答任务,任务的输入是一张图片和关于这个图片的提问，输出是对问题的回答；请给我一个关于这种任务的分析过程的CoT，这个CoT应该包含首先分析什么，然后分析什么"}
  ]
)

print(completion)  # 响应
print(completion.choices[0].message)
'''