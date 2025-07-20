import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image
import requests
from PIL import Image
import torch.nn as nn

model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto", 
)
processor = LlavaOnevisionProcessor.from_pretrained(model_id)

conversation = [
    {

      "role": "user",
      "content": [
          #{"type": "text","text":"图片中的人是男性还是女性？为什么呢？请注意以下几点，以便更好地回答问题："},
          {"type": "text","text":"这张图片中的人是男性还是女性? A未知 B男性 C女性"},
          #{"type": "text","text":"托盘的左边是什么？请注意以下提示方便更好回答问题：1、确定参考框架，是托盘的左侧，而非画面的左侧 2、确定托盘上的物品 "},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)



#JPG
raw_image = Image.open("/data/aovkqa/train2017/000000012927.jpg").convert("RGB")


'''
#URL
image_file = "https://th.bing.com/th/id/OIP.pTU-7xHh8y_HmHihLlI4vwHaFf?r=0&o=7rm=3&rs=1&pid=ImgDetMain"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
'''

'''
选择题
'''
inputs = processor(
    images=raw_image, 
    text=prompt, 
    return_tensors='pt'
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 移动到模型设备

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )
    hidden_states = outputs.hidden_states
    hidden_states = hidden_states[-1] 
    hidden_states = hidden_states[:,-1,:]
    linear_layer = nn.Linear(3584, 3)
    linear_layer.to(hidden_states.device)
    logits = linear_layer(hidden_states)
    print(f"logits shape: {logits.shape}")
    print(f"logits: {logits}")


'''
正常输出
inputs = processor(
    images=raw_image, 
    text=prompt, 
    return_tensors='pt')
# Move input tensors to the same device as the model
output = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
print("Here is the prompt",prompt)
print(processor.decode(output[0][2:], skip_special_tokens=True))
print(inputs.keys())
'''
