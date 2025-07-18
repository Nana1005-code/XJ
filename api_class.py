import os
import base64
from openai import OpenAI

# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-5MvaVcI62IFzE0gbFDaDYeOksJsoxKOHcHs5JC9NFkuTGRSZ"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

def generate_tips_from_image_and_question(image_path, question):
    """
    调用 OpenAI API 根据图片和提问生成推理时需要关注的关键点提示。

    参数:
    image_path (str): 图片文件的路径
    question (str): 用户的提问

    返回:
    list: 返回生成的提示列表
    """
    # 将图片转换为 base64 编码
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{image_base64}"

    # 构造 API 请求内容
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"针对这张图片和提问，请输出推理时需要关注的关键点提示。"
                            f"要求："
                            f"1. 用简洁的短句描述关键点；"
                            f"2. 按照1、2、3编号输出；"
                            f"3. 不要解释，不要展开，不输出答案,不输出具体行为或现象描述。"
                            f"提问：{question}"
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

    # 处理返回的响应，获取提示列表
    api_tips_text = response.choices[0].message.content.strip()
    api_tips = api_tips_text.splitlines()
    return api_tips

if __name__ == "__main__":
    # 测试函数
    image_path = "./1752660927641.jpg"  # 替换为实际的图片路径
    question = "这张图片中谁处于危险？"
    tips = generate_tips_from_image_and_question(image_path, question)
    print("生成的提示：")
    for tip in tips:
        print(tip)