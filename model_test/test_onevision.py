import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image
import requests
from PIL import Image

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

inputs = processor(
    images=raw_image, 
    text=prompt, 
    return_tensors='pt',
    text_kwargs={"return_mm_token_type_ids": True})
# Move input tensors to the same device as the model
#output = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
#print(processor.decode(output[0][2:], skip_special_tokens=True))
print(inputs.keys())