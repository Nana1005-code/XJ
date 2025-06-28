import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaOnevisionProcessor,LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image
import requests
from PIL import Image
from aligner import SequenceAligner

model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
models = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto", 
)
processor = LlavaOnevisionProcessor.from_pretrained(model_id)
print(type(processor))

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text","text":"哪个物体处于最危险的境地？"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)



#JPG
raw_image = Image.open("/data/aovkqa/train2017/000000012993.jpg").convert("RGB")


'''
#URL
image_file = "https://th.bing.com/th/id/OIP.pTU-7xHh8y_HmHihLlI4vwHaFf?r=0&o=7rm=3&rs=1&pid=ImgDetMain"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
'''
# 获取 <image> token 的 ID
image_token = processor.image_token  # 默认是 "<image>"
#image_token_id = processor.image_token_id  # int类型的token id

'''
拿出text_embedding
在这个位置要找出非<image>token的文本嵌入
'''
input_ids = processor.tokenizer(prompt, return_tensors='pt').input_ids.to("cuda:0")
print(input_ids.shape) #torch.Size([1, 18]) [batchsize,token_num]
input_ids = input_ids.squeeze(0)
text_embeds = models.language_model.get_input_embeddings()(input_ids)
text_embeds = text_embeds.squeeze(0)  # 去掉第0维（batch_size）


print(text_embeds)
print(text_embeds.shape) #torch.Size([18, 3584]) [batchsize,token_num,embedding_dim]
print("--"*20)

# 若 image_token_id 缺失，则补上
if not hasattr(processor, "image_token_id"):
    processor.image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    print("image_token_id manually set to:", processor.image_token_id)

# 2. 获取 <image> token id 和文本长度（注意：这一步可以结合 text_embeds 的长度确认）
image_token_id = processor.image_token_id
text_token_len = text_embeds.shape[0]  # e.g., 18

# 3. 找到input_ids中<image>的索引
image_token_index = (input_ids == image_token_id).nonzero(as_tuple=True)[0].item()

# 4. 构造等长的<image> token list，替换原来的1个<image> token
repeated_image_tokens = torch.full((text_token_len,), image_token_id, dtype=torch.long, device=input_ids.device)

# 5. 构造新的input_ids：保留<image>前后的部分，中间插入多个<image>
input_ids = torch.cat([
    input_ids[:image_token_index],
    repeated_image_tokens,
    input_ids[image_token_index + 1:]
], dim=0).unsqueeze(0)  # 还原batch维度

print("Modified input_ids shape:", input_ids.shape)  # 应为 [1, text_len * 2]

'''
拿出vision_embedding
'''
inputs = processor(images=raw_image, text=prompt, return_tensors='pt')
image_embeds = models.get_image_features(
    pixel_values=inputs["pixel_values"],
    image_sizes=torch.tensor(inputs["image_sizes"]),vision_feature_layer=-1,vision_feature_select_strategy="full"
)
#print(image_embeds)
print(image_embeds[0].shape) #torch.Size([3, 729, 3584]) (num_patches, image_length, embed_dim)

print("**"*50)
image_embeds = torch.cat(image_embeds, dim=0)

#print(image_embeds)
print(image_embeds.shape) #torch.Size([729, 3584])

print("##"*50)
image_embeds = image_embeds.reshape(-1, image_embeds.size(-1))
#print(image_embeds)
print(image_embeds.shape) #[2187,3584]

'''
序列对齐模块
'''
# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 初始化对齐模块
top_k = 1000
aligner = SequenceAligner(topk=top_k).to("cuda:0")

inputids_new = input_ids.clone()
inputids_new = inputids_new.squeeze(0)
is_image_token = (inputids_new[0] == image_token_id)

aligned_input = aligner(image_embeds, text_embeds,is_image_token)
#print(aligned_input)
#print(aligned_input.shape)

'''
对齐之后输入到language_model
'''
aligned_input = aligned_input.unsqueeze(0).to("cuda:0") #增加batch维度
print(input_ids.shape)
#input_ids = torch.cat([input_ids, input_ids], dim=1)######<image>token
print(input_ids.shape)

generated_ids = models.generate(
    inputs_embeds=aligned_input,
    input_ids=input_ids,  # prompt 的 token ids
    max_new_tokens=20,
)
answer = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated answer:", answer)

'''
代码修改逻辑：
1、找到Input_id对应的<image>
2、prompt中存在一个<image>,需要在这个位置将其扩展为与文本等长的<image>
3、序列进行拼接的时候,需要考虑到将视觉的序列拼接到input_id中对应的<image>位置.而不是简单的V,X
'''

'''
注意力模块
'''

