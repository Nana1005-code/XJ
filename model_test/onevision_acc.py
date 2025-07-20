import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from aokvqa_dataset import AOKVQADataset 
from torchvision import transforms
from torch.utils.data import DataLoader

model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto", 
)
processor = LlavaOnevisionProcessor.from_pretrained(model_id)
linear_layer = nn.Linear(3584, 4)
#计算得分
def generate_answer(model, inputs):
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        hidden_states = hidden_states[-1] 
        hidden_states = hidden_states[:,-1,:]
        linear_layer.to(hidden_states.device)
        logits = linear_layer(hidden_states)
        #print(f"logits shape: {logits.shape}")
        #print(f"logits: {logits}")
        predicted_index = torch.argmax(logits, dim=-1)
        #print(f"predicted_index: {predicted_index.item()}")
        del outputs, hidden_states
        torch.cuda.empty_cache()
        return predicted_index.item(),logits

def prepare_dataloader(jsonl_path, batch_size=2):
    # 设置图像预处理（例如：调整大小、转为 tensor、标准化）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建 Dataset 实例
    dataset = AOKVQADataset(jsonl_path=jsonl_path, image_transform=transform)

    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader

def test(model, dataloader, device=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): #不需要计算梯度
        num=0
        for batch in tqdm(dataloader):
            #print(batch)
            images=batch['image_path']
            for i in range(len(batch['image_path'])):
                question = batch["question"][i]
                choices = batch["choices"][i]
                text = f"{question} Answer:{' '.join(choices)}"
                labels = torch.Tensor(batch['label'][i]).to(device)
                #print(labels)
                prompt = f"Question:{text}"
                conversation = [
                {

                    "role": "user",
                    "content": [
                    {"type": "text","text":prompt},
                    {"type": "image"},
                        ],
                    },
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                image=images[i]
                raw_image = Image.open(image).convert("RGB")
                inputs = processor(
                    images=raw_image, 
                    text=prompt, 
                    return_tensors='pt'
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 移动到模型设备

                predicted_label,logits = generate_answer(model, inputs)
            
            # predicted_label 应为 tensor 或 numpy
                if isinstance(predicted_label, torch.Tensor):
                    predicted_label = predicted_label.cpu()
                correct += (predicted_label == labels.cpu()).sum().item()
                total += 1
                #print("done 1")
            num+=1
            if num==100:
                break
        acc = correct / total if total > 0 else 0
        return acc

if __name__ == "__main__":
    """
    Test
    """
    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    #model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    
    # 使用 GPU
    device = "cuda"
   
    # 准备数据集
    dataloader = prepare_dataloader(jsonl_path="process.json")

    cor=test(model, dataloader, device=device)
    print(cor)