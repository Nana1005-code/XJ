import os
import base64
import json
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
'''
对于json文件的每一张图片和问题，生成提示CoT
最后形成一个json文件
'''

# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-qh2eHpJ3jIUzbQ8VCWZsyHvkDWRfn1kUbCIUJx38PuFdkyp4"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://chatapi.zjt66.top/v1"
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
        model="qwen2.5-vl-72b-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"针对这张图片和提问，请输出推理时需要关注的关键点提示。"
                            f"要求："
                            f"1. 用简洁的短句描述关键点，按照1、2、3编号输出；"
                            f"2. 不要解释，不要展开，不输出答案,不输出具体行为或现象描述。"
                            f"3. 用英文回答。"
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
    input_path = "/home/user/2024_xj/XJ/process_cot.json"
    output_path = "/home/user/2024_xj/XJ/process_with_tips.json"

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        lines = fin.readlines()
        for line in tqdm(lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            image_path = data.get("image_path")
            question = data.get("question")
            tips_list = generate_tips_from_image_and_question(image_path, question)
            #print(tips_list)
            # 将提示列表合并为字符串，如 "1、2、3、"
            tips_text = " ".join(tips_list)
            #tips_str = "、".join([tip.strip() for tip in tips_list if tip.strip()]) + "、"
            data["tips"] = tips_text
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
   
    '''
    image_path = "./1752660927641.jpg"  # 替换为实际的图片路径
    question = "这张图片中谁处于危险？"
    tips = generate_tips_from_image_and_question(image_path, question)
    print("生成的提示：")
    for tip in tips:
        print(tip)
    '''
