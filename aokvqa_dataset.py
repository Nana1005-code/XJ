import json
from PIL import Image
from torch.utils.data import Dataset

class AOKVQADataset(Dataset):
    #返回一个字典
    def __init__(self, jsonl_path, image_transform=None):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        self.image_transform = image_transform

    def procee_sample(self):
        for i in range(len(self.samples)):
            sample = self.samples[i]
            image = Image.open(sample["image_path"]).convert("RGB")
            #if self.image_transform:
                #image = self.image_transform(image)

            question = sample["question"]
            choices = sample["choices"]
            text = f"{question} Answer:{' '.join(choices)}"
            label = sample["label"]
            data_item={
            "image": image,
            "text": text,
            "label": label
            }
            self.data.append(data_item)
        # 拼接问题+选项
        


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
