import torch
import numpy as np
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
from aligner import SequenceAligner
from api_class import generate_tips_from_image_and_question  # 调用封装好的 API
import torch.nn as nn
import torch.nn.functional as F
'''
更改生成方式的test，已跑通
'''

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

    def get_classification_scores(self, text_embeds):
        """
        使用线性层将文本嵌入转化为4个选项的得分。
        text_embeds: 模型输出的文本嵌入（假设维度为 [batch_size, 3584]）
        """
        logits = self.linear_layer(text_embeds)  # [batch_size, 4]
        return logits

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
        input_ids = self.processor.tokenizer(prompt, return_tensors='pt',padding=True).input_ids.to(self.device)
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.squeeze(0) #[18,3584]
        #print("-----打印text_embeds------",text_embeds)
        return input_ids, text_embeds

    def expand_image_tokens(self, input_ids, text_embeds):
        input_ids=input_ids.squeeze(0)
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
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt',padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()} # 显式移动到 GPU
        image_embeds = self.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_sizes=torch.tensor(inputs["image_sizes"]),
            vision_feature_layer=-1,
            vision_feature_select_strategy="full"
        )
        image_embeds=torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.reshape(-1, image_embeds.size(-1))
        #print("-----打印image_embeds------",image_embeds)
        #print(image_embeds.shape)
        return image_embeds

    def align_sequences(self, image_embeds, text_embeds, is_image_token, top_k=1000):
        torch.manual_seed(42)
        aligner = SequenceAligner(topk=top_k).to(self.device)
        aligned_input = aligner(image_embeds, text_embeds, is_image_token)
        return aligned_input

    def generate_answer(self, aligned_input, input_ids, max_new_tokens=20):
        #拓展batch维度，并做深拷贝，复制
        aligned_input_class=aligned_input.unsqueeze(0).detach().clone() #[1,145,3584]
        with torch.no_grad():#无梯度推理
            aligned_input = aligned_input.unsqueeze(0)
            # 1. 获取隐藏层状态
            outputs = self.model(
                inputs_embeds=aligned_input,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states  # list of layers
            # 2. 打印隐藏层状态的数量和每一层的形状
            #print(f"隐藏层数量: {len(hidden_states)}")
            #for i, h in enumerate(hidden_states):
                #print(f"第{i}层 shape: {h.shape}")
            
            #拿出最后一层
            hidden_states = hidden_states[-1] #[1,145,3584] 
            #print(f"最后一层 shape: {hidden_states.shape}")
            #拿出H1
            hidden_states = hidden_states[:,-1,:]
            #print(f"第一行 shape: {hidden_states.shape}")
            self.linear_layer.to(hidden_states.device)
            logits = self.linear_layer(hidden_states)
            #print(f"logits shape: {logits.shape}")
            #print(f"logits: {logits}")
            logits=torch.softmax(logits, dim=-1) 
            predicted_index = torch.argmax(logits, dim=-1)
            #print(f"predicted_index: {predicted_index.item()}")
            return predicted_index.item(),logits
        

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
    生成答案，并计算每个选项的得分，代替了之前generate answer的方法。
    """
    def generate_answer_and_score(self, aligned_input, input_ids, max_new_tokens=20):

        aligned_input = aligned_input.unsqueeze(0).to(self.device)
        
        # 获取生成的文本嵌入
        output_ids = self.model.generate(
            inputs_embeds=aligned_input,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        # 解码生成的答案，和之前的相同
        answer = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return answer

    def run(self, image_path, question, api_tips):
        augmented_question = self.build_augmented_question(question, api_tips)
        prompt = self.build_prompt(augmented_question)
        print("Here is the final question:", prompt)
        input_ids, text_embeds = self.get_text_embeds(prompt)
        input_ids, is_image_token = self.expand_image_tokens(input_ids, text_embeds)
        image_embeds = self.get_image_embeds(image_path, prompt)
        aligned_input = self.align_sequences(image_embeds, text_embeds, is_image_token)
        answer = self.generate_answer(aligned_input, input_ids, max_new_tokens=20)
        return answer



if __name__ == "__main__":
    

    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    image_path = "/data/aovkqa/train2017/000000012991.jpg"
    question = "这张图片中的人是男性还是女性? A未知 B男性 C女性"

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