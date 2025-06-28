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

'''
数据集预处理
'''

#创建模型
print("-----------------create model-------------------------")
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

#print(model)
#for name, parameter in model.named_parameters():
    #print(name)


#Lora
print("-----------------Lora begin-------------------------")
config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                    target_modules=".*\.1.*query_key_value", #指定插入LoRA adapter 的模块
                    modules_to_save=["word_embeddings"]) #不插入 LoRA adapter 但需要保存/训练的原始模块

#创建PEFT模型
print("-----------------create peft model-------------------------")
model = get_peft_model(model, config) #只是得到一个微调模型，未将微调模型融入原本模型
#print(model)

#model.print_trainable_parameters() #打印微调后需要训练的参数 peft专用

#开始训练 配置训练参数,会得到新的模型权重
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
