import torch
import numpy as np
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
from aligner import SequenceAligner

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
        print("-----打印text_embeds------",text_embeds)
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
        print(new_input_ids)
        return new_input_ids, is_image_token

    def get_image_embeds(self, image_path, prompt):
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt')
        image_embeds = self.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_sizes=torch.tensor(inputs["image_sizes"]),
            vision_feature_layer=-1,
            vision_feature_select_strategy="full"
        )
        image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.reshape(-1, image_embeds.size(-1))
        print("-----打印image_embeds------",image_embeds)
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

    def run(self, image_path, question):
        prompt = self.build_prompt(question)
        input_ids, text_embeds = self.get_text_embeds(prompt)
        input_ids, is_image_token = self.expand_image_tokens(input_ids, text_embeds)
        image_embeds = self.get_image_embeds(image_path, prompt)
        aligned_input = self.align_sequences(image_embeds, text_embeds, is_image_token)
        print("-----打印aligned_input------",aligned_input.shape)
        answer = self.generate_answer(aligned_input, input_ids, max_new_tokens=100)
        return answer

if __name__ == "__main__":
    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    image_path = "/data/aovkqa/train2017/000000012991.jpg"
    question = "图片中的人是什么性别？"

    pipeline = LlavaOnevisionPipeline(model_id)
    answer = pipeline.run(image_path, question)
    print("Generated answer:", answer)
