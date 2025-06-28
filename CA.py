'''
校准网络
'''
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

device = "cuda:1"

class CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, qas, labels, scores1, scores2):
        self.qas = qas
        self.labels = labels
        self.scores1 = scores1
        self.scores2 = scores2
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, idx):
        qa_list = self.qas[idx].split('\n')[:-1]
        qa_clean = [qa_list[0]] + [i for i in qa_list[1:] if i.strip()]
        qa = '[CLS]'.join(qa_clean)
        inputs = self.tokenizer(qa, truncation=True, max_length=256, padding='max_length', return_tensors='pt').to(device)
        label = torch.tensor(self.labels[idx]).to(device)

        s1 = [float(i) for i in eval(self.scores1[idx])]  # p(y|q)
        s2 = [float(i) for i in eval(self.scores2[idx])]  # p(y|q,v,s,a)
        scores = torch.tensor(s1 + s2).to(device)  # 拼接为8维分布

        return inputs, scores, label

class BertCalibrationNet(nn.Module):
    def __init__(self, dropout=0.18):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 4)  # 分类为4个选项
        self.linear3 = nn.Linear(8, 1)    # 8维注意力加权
        self.linear4 = nn.Linear(768, 1)  # 可选：用于MSE监督
        self.relu = nn.ReLU()

    def forward(self, inputs, scores):
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        output = self.bert(input_ids, attention_mask=attention_mask)
        cls_index = (input_ids == 101).nonzero().view(-1, 5, 2)[:, 1:, :]

        # 提取4个[CLS]位置对应的输出
        option_reps = output.last_hidden_state[cls_index[:, :, 0], cls_index[:, :, 1], :]  # [B, 4, 768]
        option_reps_dropout = self.dropout(option_reps)
        reps_expand = option_reps_dropout.repeat(1, 1, 2).view(-1, 8, 768)  # [B, 8, 768]
        scores = scores.unsqueeze(1).repeat(1, 1, 768).view(-1, 8, 768)     # [B, 8, 768]

        attn_output = torch.mul(reps_expand, scores).transpose(1, 2)       # [B, 768, 8]
        attn_output = self.linear3(attn_output).squeeze(-1)                # [B, 768]
        attn_output = self.relu(attn_output)
        attn_output = self.dropout(attn_output)

        hidden = self.relu(self.linear1(attn_output))
        logits = self.linear2(hidden)

        # 用于MSE loss的输出（可选）
        probs = F.softmax(self.linear4(option_reps).squeeze(-1), dim=-1)

        return logits, probs

# 示例训练流程（核心部分）
def train(model, dataloader, optimizer, use_mse=False):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    model.train()
    for batch in dataloader:
        inputs, scores, labels = batch
        outputs, probs = model(inputs, scores)
        loss = ce_loss(outputs, labels)
        if use_mse:
            ave_tensor = torch.ones_like(probs) / probs.size(1)
            loss += 0.1 * mse_loss(probs, ave_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
