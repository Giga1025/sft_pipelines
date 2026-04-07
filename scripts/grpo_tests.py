import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import wandb

# ─── 1. Configuration ────────────────────────────────────────────────────────
MODEL_ID = "/data01/yaswanthreddy/models/mistral-7b-v0.1" # Using the base model as requested
OUTPUT_DIR = "./mistral-grpo-base-test"
GROUP_SIZE = 4  # The "Group" in GRPO. Generates 4 responses per prompt to calculate the baseline.

# ─── 2. Reward Functions ─────────────────────────────────────────────────────
# In GRPO, you don't need a critic model. You just need functions that take 
# the generated completions and return a float score. TRL allows you to stack them.

def format_reward_func(completions, **kwargs):
    """
    Reward 1: Partial Formatting (Scaffolding).
    Gives points for individual tags to help the base model discover the format.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        
        # 1. Check for individual tags (0.2 each)
        if "<think>" in completion: score += 0.2
        if "</think>" in completion: score += 0.2
        if "<answer>" in completion: score += 0.2
        if "</answer>" in completion: score += 0.2
        
        # 2. Check for the correct sequence/order (0.2 bonus)
        # Regex: think starts, then ends, then answer starts, then ends.
        pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
        if re.search(pattern, completion, re.DOTALL | re.IGNORECASE):
            score += 0.2
            
        rewards.append(score)
    return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward 2: Accuracy.
    Extracts the text inside <answer>...</answer> and compares it to the ground truth.
    """
    rewards = []
    # 'answer' here is the ground truth passed from the dataset
    for comp, truth in zip(completions, answer):
        # Extract the final answer from the model's generation
        match = re.search(r"<answer>(.*?)</answer>", comp, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip()
            # Simple string match (you can make this more robust for math/code)
            if extracted_answer == str(truth).strip():
                rewards.append(2.0) # Higher weight for actually getting it right
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ─── 3. Dataset Preparation ──────────────────────────────────────────────────
# TRL's GRPO expects a dataset with at least a 'prompt' column. 
# Any other columns (like 'answer') are passed to the reward functions as **kwargs.

def format_dataset_for_grpo(example):
    # We provide a strict system prompt to kickstart the formatting behavior
    system_prompt = (
        "You are a logical reasoning assistant. First, think step-by-step inside "
        "<think>...</think> tags. Then, output your final concise answer inside "
        "<answer>...</answer> tags."
    )
    user_prompt = example["question"]
    
    # We construct the prompt exactly how Mistral expects it
    prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    
    return {
        "prompt": prompt,
        "answer": example["answer"] # Passed to the correctness_reward_func
    }

def filter_by_length(example):
    # Tokenize the prompt without adding special tokens (since we already added <s>)
    inputs = tokenizer(example["prompt"], add_special_tokens=False)
    return len(inputs["input_ids"]) <= 256

print("Loading and formatting dataset...")
# Using GSM8K (a math word problem dataset) as a perfect testbed for reasoning
dataset = load_dataset("gsm8k", "main", split="train[:500]") 
# GSM8K answers have a specific format, we extract just the final number
dataset = dataset.map(lambda x: {"answer": x["answer"].split("#### ")[-1]})
train_dataset = dataset.map(format_dataset_for_grpo, remove_columns=dataset.column_names)

train_dataset = train_dataset.filter(filter_by_length)

# ─── 4. Model & LoRA Setup ───────────────────────────────────────────────────

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}, # Load on GPU 0, adjust if you have multiple GPUs
    use_cache=False # Crucial for gradient checkpointing
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─── 5. GRPO Training ────────────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5, # GRPO learning rates should be lower than SFT (e.g., 1e-5 to 5e-6)
    per_device_train_batch_size=1, # Keep this at 1, the effective BS is determined by num_generations
    gradient_accumulation_steps=4,
    optim = "adamw_bnb_8bit",
    max_steps=50,
    bf16=True,
    logging_steps=5,
    save_steps=50,
    report_to="wandb",
    gradient_checkpointing=True,
    use_liger_kernel=True,
    
    # GRPO Specific Parameters
    num_generations=GROUP_SIZE,    # How many responses to generate per prompt
    max_completion_length=512,     # Give it enough room to "think"
    beta=0.04,                     # KL penalty coefficient (how strictly to tether to the Reference model)
    
    # If you install vLLM (`pip install vllm`), set this to True for a MASSIVE generation speedup
    use_vllm=False, 
)


wandb.init(
                project="mistral_Grpo",
                name="50steps-trl-grpo",
                resume="allow",
                dir="/home/yaswanthreddy/snorkel/monitoring",
            )

print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[format_reward_func, correctness_reward_func],
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("Starting GRPO Training...")
trainer.train()
wandb.finish()
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print("Training complete!")