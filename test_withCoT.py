import torch
import numpy as np
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
from aligner import SequenceAligner
from api_class import generate_tips_from_image_and_question  # 调用封装好的 API
import torch.nn as nn
import torch.nn.functional as F

class LlavaOnevisionPipeline:
    def __init__(self, model_id, device="cuda:0"):
        self.device = device
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        )
        self.processor = LlavaOnevisionProcessor.from_pretrained(model_id)
        self.image_token = self.processor.image_token
        if not hasattr(self.processor, "image_token_id"):
            self.processor.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_token_id = self.processor.image_token_id
        #new\\\\\\\\\\\\\\\\\\\\\\\\\\
        # 定义一个线性层来映射模型输出到4个选项的得分
        self.linear_layer = nn.Linear(3584, 4)  # 3584 是假设模型输出的维度，4 是选项的个数

    def build_prompt(self, question):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        return prompt

    def get_text_embeds(self, prompt):
        input_ids = self.processor.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        input_ids = input_ids.squeeze(0)
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.squeeze(0) #[18,3584]
        #print("-----打印text_embeds------",text_embeds)
        return input_ids, text_embeds

    def expand_image_tokens(self, input_ids, text_embeds):
        text_token_len = text_embeds.shape[0]
        is_image_token = (input_ids == self.image_token_id) #哪些位置是图像标记
        image_token_index = is_image_token.nonzero(as_tuple=True)[0].item()
        repeated_image_tokens = torch.full((text_token_len,), self.image_token_id, dtype=torch.long, device=input_ids.device)
        new_input_ids = torch.cat([
            input_ids[:image_token_index],
            repeated_image_tokens,
            input_ids[image_token_index + 1:]
        ], dim=0).unsqueeze(0)
        #print(new_input_ids)
        return new_input_ids, is_image_token

    def get_image_embeds(self, image_path, prompt):
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt')
        inputs = {k: v.to("cuda") for k, v in inputs.items()}  # 显式移动到 GPU
        image_embeds = self.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_sizes=torch.tensor(inputs["image_sizes"]),
            vision_feature_layer=-1,
            vision_feature_select_strategy="full"
        )
        image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.reshape(-1, image_embeds.size(-1))
        #print("-----打印image_embeds------",image_embeds)
        return image_embeds

    def align_sequences(self, image_embeds, text_embeds, is_image_token, top_k=1000):
        torch.manual_seed(42)
        aligner = SequenceAligner(topk=top_k).to(self.device)
        aligned_input = aligner(image_embeds, text_embeds, is_image_token)
        return aligned_input

    def generate_answer(self, aligned_input, input_ids, max_new_tokens=20):
        aligned_input = aligned_input.unsqueeze(0).to(self.device)
        output_ids = self.model.generate(
            inputs_embeds=aligned_input,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        answer = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return answer

    def build_augmented_question(self, question: str, api_tips: list[str]) -> str:
        """
        拼接原始问题和 API 返回的提示词。
        """
        tips_text = " ".join(api_tips)
        augmented = (
            f"{question}，并告诉我判断的理由。请注意以下提示以便更好地回答问题："
            f"{tips_text}"
        )
        return augmented
    

    """
    生成答案，并计算每个选项的得分。
    """
    def generate_answer_and_score(self, aligned_input, input_ids, max_new_tokens=20):

        aligned_input = aligned_input.unsqueeze(0).to(self.device)
        
        # 获取生成的文本嵌入
        output_ids = self.model.generate(
            inputs_embeds=aligned_input,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        # 获取文本嵌入
        text_embeds = self.model.language_model.get_input_embeddings()(output_ids.squeeze(0))
        
        # 使用线性层将文本嵌入转化为4个选项的得分
        logits = self.get_classification_scores(text_embeds)
        
        # 使用 softmax 计算概率
        probabilities = F.softmax(logits, dim=-1)  # [batch_size, 4]
        
        # 获取得分最高的选项
        predicted_label = torch.argmax(probabilities, dim=-1).item()
        
        # 解码生成的答案
        answer = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return answer, predicted_label, probabilities

    def run(self, image_path, question, api_tips):
        augmented_question = self.build_augmented_question(question, api_tips)
        prompt = self.build_prompt(augmented_question)
        print("Here is the final question:", prompt)
        input_ids, text_embeds = self.get_text_embeds(prompt)
        input_ids, is_image_token = self.expand_image_tokens(input_ids, text_embeds)
        image_embeds = self.get_image_embeds(image_path, prompt)
        aligned_input = self.align_sequences(image_embeds, text_embeds, is_image_token)
        answer = self.generate_answer(aligned_input, input_ids, max_new_tokens=1000)
        return answer
        
        '''
        带CoT
        augmented_question = self.build_augmented_question(question, api_tips)
        prompt = self.build_prompt(augmented_question)
        print("Here is the final question:", prompt)
        input_ids, text_embeds = self.get_text_embeds(prompt)
        input_ids, is_image_token = self.expand_image_tokens(input_ids, text_embeds)
        image_embeds = self.get_image_embeds(image_path, prompt)
        aligned_input = self.align_sequences(image_embeds, text_embeds, is_image_token)
        answer = self.generate_answer(aligned_input, input_ids, max_new_tokens=1000)
        return answer
        '''
        
        '''
        初始
        prompt = self.build_prompt(question)
        input_ids, text_embeds = self.get_text_embeds(prompt)
        input_ids, is_image_token = self.expand_image_tokens(input_ids, text_embeds)
        image_embeds = self.get_image_embeds(image_path, prompt)
        aligned_input = self.align_sequences(image_embeds, text_embeds, is_image_token)
        #print("-----打印aligned_input------",aligned_input.shape)
        answer = self.generate_answer(aligned_input, input_ids, max_new_tokens=100)
        return answer
        '''



if __name__ == "__main__":
    

    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    image_path = "/data/aovkqa/train2017/000000012991.jpg"
    question = "这张图片中的人是男性还是女性？"

    # 调用 API 获取提示
    #api_tips = generate_tips_from_image_and_question(image_path, question)
    api_tips = [
        "1. 人物身材曲线，肤色和面部特征。","2. 人物穿着搭配，衣服款色，颜色和材质。","3. 近一步分析人物发型和配饰。"
    ]

    # 运行 pipeline
    pipeline = LlavaOnevisionPipeline(model_id)
    answer = pipeline.run(image_path, question, api_tips)

    print("Generate answer:", answer)

    '''
    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    image_path = "/data/aovkqa/train2017/000000012991.jpg"
    question = "图片中的人是什么性别？"
    pipeline = LlavaOnevisionPipeline(model_id)  # 添加缺失的实例化
    answer = pipeline.run(image_path, question)
    print("Generate answer:", answer)
    '''