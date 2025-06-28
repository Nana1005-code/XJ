#导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

#加载数据集
ds = Dataset.load_from_disk("../Lora/data/alpaca_data_zh/")
#print(ds[:3]) #打印输出

#tokenization
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
#print(tokenizer)

#数据集预处理，将其变成模型能够处理的格式
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
print("hello")
print(tokenized_ds) #这里有报错为什么不出现处理好的整个数据集
print("here")

tokenizer.decode(tokenized_ds[1]["input_ids"]) #decode方法会处理token IDs 形成一个可读的字符串输出
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"])))

#创建模型
print("-----------------create model-------------------------")
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

#print(model)
#for name, parameter in model.named_parameters():
    #print(name)


#Lora
print("-----------------Lora begin-------------------------")
config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                    target_modules=".*\.1.*query_key_value", 
                    modules_to_save=["word_embeddings"])

#创建PEFT模型
print("-----------------create peft model-------------------------")
model = get_peft_model(model, config) #只是得到一个微调模型，未将微调模型融入原本模型
#print(model)

#model.print_trainable_parameters() #打印微调后需要训练的参数 peft专用

#开始训练 配置训练参数
args = TrainingArguments(
    output_dir="./LORA",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    use_cpu=True
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

#模型训练
print("-----------------Training-------------------------")
print(model.device)
trainer.train()

#model = model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
