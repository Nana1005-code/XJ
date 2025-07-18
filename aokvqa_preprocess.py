import json
import os
from tqdm import tqdm

def preprocess_aokvqa_train(
    json_path: str,
    coco_img_root: str,
    save_path: str = None
):
    """
    预处理 AOKVQA train.json 数据
    输出格式: list of dict
    每个样本包含:
        - image_path: COCO 图像路径
        - question: 问题文本
        - choices: 候选答案列表
        - label: 正确答案的下标
    """
    # 加载 json,只读模式打开
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_samples = [] #存储处理后的样本
     
    for item in tqdm(data, desc="Processing"): # 遍历数据，产生进度条
        image_id = item['image_id']
        question = item['question']
        choices = item['choices']
        label = item['correct_choice_idx']

        # COCO 图片路径（12位数字+jpg）
        image_filename = f"{image_id:012d}.jpg"
        image_path = os.path.join(coco_img_root, image_filename)

        # 检查图片是否存在
        if not os.path.isfile(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        sample = {
            "image_path": image_path,
            "question": question,
            "choices": choices,
            "label": label
        }

        processed_samples.append(sample)

    print(f"✅ Processed {len(processed_samples)} samples.")

    # 可选：保存成 jsonl 或 pickle
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as out_f:
            for sample in processed_samples:
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"✅ Saved to {save_path}")

    return processed_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess AOKVQA train.json")
    parser.add_argument("--json_path", type=str, required=True, help="Path to train.json")
    parser.add_argument("--coco_img_root", type=str, required=True, help="Path to COCO train2017 image folder")
    parser.add_argument("--save_path", type=str, default="aokvqa_train_processed.jsonl", help="Path to save preprocessed data")
    args = parser.parse_args()

    preprocess_aokvqa_train(
        json_path=args.json_path,
        coco_img_root=args.coco_img_root,
        save_path=args.save_path
    )
