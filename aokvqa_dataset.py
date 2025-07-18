class AOKVQADataset(Dataset):
    #返回一个字典
    def __init__(self, jsonl_path, image_transform=None):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        question = sample["question"]
        choices = sample["choices"]
        label = sample["label"]

        # 拼接问题+选项
        text = f"{question} Answer:{' '.join(choices)}"

        return {
            "image": image,
            "text": text,
            "label": label
        }
