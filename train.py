import torch
from torch.utils.data import DataLoader
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from attention import compute_attention_alignment_loss  # 使用你提供的 loss 计算函数
from aokvqa_dataset import AOKVQADataset  # 你的自定义 Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from test_old import LlavaOnevisionPipeline

'''
在test withcot里面先跑一个看看有没有bug，run还没写，线性层矩阵维度还没改
注意力的理论再推一次
API测试一下
先一个batch里面开始训练跑跑bug，打印一下各个情况，看看情况
'''

def setup_lora(model, target_layers=[15, 16, 17], r=8, alpha=32, dropout=0.1):
    """
    设置 LoRA 微调，目标层为第 16、17、18 层。
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,  # 低秩矩阵的秩
        lora_alpha=alpha,  # LoRA的缩放因子
        target_modules=["q_proj", "k_proj"],  # 只修改 Q 和 K 矩阵
        lora_dropout=dropout
    )
    
    # 选择需要 LoRA 微调的层
    model = get_peft_model(model, config)
    
    # 只修改指定的层
    for name, param in model.named_parameters():
        if any(f"transformer.layers.{layer}.attention" in name for layer in target_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model

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

def train(model, dataloader, optimizer, device=None):
    model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 获取图片、问题+选项和标签
        images = batch['image'].to(device)  # 图像
        texts = batch['text']  # 问题+选项
        labels = batch['label'].to(device)  # 真实标签

        # 为每个问题+选项生成 prompt
        prompt = [f"Question:{text}" for text in texts]  # 构造问题+选项

        # 获取模型的图像和文本嵌入
        image_embeds = model.get_image_embeds(images, prompt)
        text_embeds = model.get_text_embeds(prompt)
        
        # 使用 aligner 来对齐图像和文本嵌入
        aligned_input = model.align_sequences(image_embeds, text_embeds)
        
        # 生成答案并计算得分
        answer, predicted_label, probabilities = model.generate_answer_and_score(aligned_input, text_embeds)
        
        # 计算损失
        loss_fn = torch.nn.CrossEntropyLoss()  # 使用 CrossEntropy 损失来计算标签和预测的差异
        loss = loss_fn(probabilities, labels)  # 这里的 probabilities 是 softmax 输出的得分
        
        # 提取层注意力（示例：假设你有特定的 attention 层）
        attn_map16 = model.transformer.layers[15].self_attn.attention
        attn_map17 = model.transformer.layers[16].self_attn.attention
        attn_map18 = model.transformer.layers[17].self_attn.attention

        # 获取文本的 token 数量
        n_text = texts.size(1)

        # 计算注意力均衡损失
        attn_loss16 = compute_attention_alignment_loss(attn_map16, n_text)
        attn_loss17 = compute_attention_alignment_loss(attn_map17, n_text)
        attn_loss18 = compute_attention_alignment_loss(attn_map18, n_text)

        # 合并损失
        total_loss = loss + attn_loss16 + attn_loss17 + attn_loss18

        # 反向传播
        total_loss.backward()
        optimizer.step()

        print(f"Loss: {total_loss.item()}, Predicted label: {predicted_label}, Ground truth label: {labels.item()}")

def test(pipeline, dataloader, device=None):
    pipeline.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            print(batch)
            images=batch['image_path']
            for i in range(len(batch['image_path'])):
                question = batch["question"][i]
                choices = batch["choices"][i]
                text = f"{question} Answer:{' '.join(choices)}"
                labels = torch.Tensor(batch['label'][i]).to(device)
                prompt = [f"Question:{text}"]
                conversation = [
                {

                    "role": "user",
                    "content": [
                    {"type": "text","text":prompt},
                    {"type": "image"},
                        ],
                    },
                    ]
                prompt = pipeline.processor.apply_chat_template(conversation, add_generation_prompt=True)
                image=images[i]
                image_embeds = pipeline.get_image_embeds(image, prompt)
                input_ids, text_embeds = pipeline.get_text_embeds(prompt)
                input_ids, is_image_token = pipeline.expand_image_tokens(input_ids, text_embeds)
                aligned_input = pipeline.align_sequences(image_embeds, text_embeds, is_image_token)

                predicted_label = pipeline.generate_answer(aligned_input, text_embeds)
            
            # predicted_label 应为 tensor 或 numpy
                if isinstance(predicted_label, torch.Tensor):
                    predicted_label = predicted_label.cpu()
                correct += (predicted_label == labels.cpu()).sum().item()
                total += labels.size(0)
    acc = correct / total if total > 0 else 0
    return acc

if __name__ == "__main__":
    '''
    Test
    '''
    model_id = "/data/huggingface/models/llava-hf_llava-onevision-qwen2-7b-ov-hf"
    #model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    
    # 使用 GPU
    device = "cuda"
    #model.to(device)
    pipeline = LlavaOnevisionPipeline(model_id, device=device)
   
    # 准备数据集
    dataloader = prepare_dataloader(jsonl_path="process.json")

    #开始测试
    test(pipeline, dataloader, device=device)
    '''
    train
    model_id = "llava-hf_llava-onevision-qwen2-7b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    model = setup_lora(model, target_layers=[15, 16, 17], r=8, alpha=32, dropout=0.1)  # 设置 LoRA 微调
    
    # 使用 GPU
    device = "cuda"
    model.to(device)
    
    # 准备数据集
    dataloader = prepare_dataloader(jsonl_path="train_data.jsonl")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 训练
    train(model, dataloader, optimizer, device=device)
    '''

