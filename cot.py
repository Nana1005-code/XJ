import torch
from PIL import Image
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

class CoTModel:
    def __init__(self, model_id):
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto"
        )
        self.processor = LlavaOnevisionProcessor.from_pretrained(model_id)

    def infer(self, image_path, text):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors='pt'
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}  # 显式移到 GPU

        output = self.model.generate(**inputs, max_new_tokens=2000, do_sample=False)
        return self.processor.decode(output[0][2:], skip_special_tokens=True)

# 用法示例
if __name__ == "__main__":
    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    llava_model = CoTModel(model_id)
    image_path = "/data/aovkqa/train2017/000000012991.jpg"
    text = "请根据图像和文本的内容，考虑“图片中的人是什么性别？”,生成一条思维链，不能输出解释，也不要直接回答。"
    result = llava_model.infer(image_path, text)
    print(result)