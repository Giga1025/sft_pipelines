import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from trl import SFTTrainer, SFTConfig, apply_chat_template
from datasets import load_dataset
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from peft import LoraConfig
import wandb

#Custom class and functions
from scripts.callbacker import ThroughputMonitorCallback

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model_path = "/data01/yaswanthreddy/models/mistral-7b-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_cache = False).to("cuda")
torch.cuda.synchronize() 
print(f"Model loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"                   
tokenizer.chat_template = (                        
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


dataset = load_dataset("json", data_files = "/data01/yaswanthreddy/data/ultrachat_200k_train_sft.jsonl", split = "train")
dataset = dataset.remove_columns(["prompt", "prompt_id"])
split_dataset = dataset.train_test_split(test_size=0.05)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

original_cols = train_dataset.column_names

# train_dataset = train_dataset.remove_columns(["prompt"])
train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=original_cols  # pass tokenizer as kwarg
)

eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, remove_columns=original_cols)

print(train_dataset.column_names, eval_dataset.column_names)
peft_config = LoraConfig(
    r= 16,
    lora_alpha= 32, # Should be double of r for alpha/r
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout= 0.05,
    bias = "none",
    task_type= "CAUSAL_LM"

)
training_args = SFTConfig(
    output_dir= "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-out-2",
    per_device_train_batch_size= 2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps= 30,
    learning_rate = 2e-4,
    bf16=True,
    gradient_checkpointing= True,  # saves vram by keeping check points periodically instead of all at once
    # report_to = "wandb",
    run_name = "mistral-sft-lora",
    # eval_strategy="steps",
    # logging_steps = 1,
    # eval_steps = 50,
    # save_strategy="steps",    
    # save_steps=50,   
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    max_length=2048,
    dataset_text_field= "text",
    use_liger_kernel = True

)

# wandb.init(project = "mistral_sft", name = "lora-run-3-300batch", dir = "/home/yaswanthreddy/snorkel/monitoring")

trainer = SFTTrainer(
    model = model,
    # peft_config= peft_config,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = eval_dataset,
    args = training_args,
    callbacks = [ThroughputMonitorCallback("mistral-sft-lora-single_gpu_test_noli_nogcheck")]
                # EarlyStoppingCallback(early_stopping_patience=4)]
)


start = time.time()
trainer.train()
# trainer.save_model("/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-out-2/final")
# wandb.finish()

print(f"Total time: {time.time() - start: .1f} seconds")