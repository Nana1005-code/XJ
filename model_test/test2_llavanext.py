import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto", 
)

#processor = LlavaNextVideoProcessor.from_pretrained(model_id)

print(model)