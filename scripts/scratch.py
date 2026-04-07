# ; import os
# ; import ray
# ; import ray.train as train
# ; from ray.train.torch import TorchTrainer, TorchConfig
# ; from ray.train import ScalingConfig, RunConfig, CheckpointConfig
# ; from trl import SFTTrainer, SFTConfig, apply_chat_template
# ; from datasets import load_dataset
# ; import torch, time
# ; from transformers import AutoModelForCausalLM, AutoTokenizer
# ; from peft import LoraConfig
# ; import wandb

# ; from callbacker import ThroughputMonitorCallback

# ; deepspeed_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
# ;                                 "../configs/ds_zero2.json")

# ; def train_loop_per_worker(config: dict):
# ;     """Ray Train worker function — runs on each GPU worker."""

# ;     model_path = config["model_path"]
# ;     # data_path  = config["data_path"]

# ;     # ── Model & tokenizer ────────────────────────────────────────────────────
# ;     model = AutoModelForCausalLM.from_pretrained(
# ;         model_path, torch_dtype=torch.bfloat16, use_cache=False
# ;     )

# ;     tokenizer = AutoTokenizer.from_pretrained(model_path) #keeping this just to avoid errors in trainer, not actually used since we pass the processed dataset directly to the trainer

# ;     tokenizer.pad_token       = tokenizer.eos_token
# ;     tokenizer.padding_side    = "right"
# ;     tokenizer.chat_template   = (
# ;         "{{ bos_token }}"
# ;         "{% for message in messages %}"
# ;         "{% if message['role'] == 'user' %}"
# ;         "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
# ;         "{% elif message['role'] == 'assistant' %}"
# ;         "{{ message['content'] + eos_token }}"
# ;         "{% endif %}"
# ;         "{% endfor %}"
# ;     )

# ;     # # ── Dataset ───────────────────────────────────────────────────────────────
# ;     # dataset = load_dataset("json", data_files=data_path, split="train")
# ;     # dataset = dataset.remove_columns(["prompt", "prompt_id"])
# ;     # split   = dataset.train_test_split(test_size=0.05)

# ;     # original_cols = split["train"].column_names
# ;     # train_dataset = split["train"].map(
# ;     #     apply_chat_template,
# ;     #     fn_kwargs={"tokenizer": tokenizer},
# ;     #     remove_columns=original_cols,
# ;     # )
# ;     # eval_dataset = split["test"].map(
# ;     #     apply_chat_template,
# ;     #     fn_kwargs={"tokenizer": tokenizer},
# ;     #     remove_columns=original_cols,
# ;     # )

# ;     # ── LoRA ──────────────────────────────────────────────────────────────────
# ;     peft_config = LoraConfig(
# ;         r=16,
# ;         lora_alpha=32,
# ;         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
# ;                         "gate_proj", "up_proj", "down_proj"],
# ;         lora_dropout=0.05,
# ;         bias="none",
# ;         task_type="CAUSAL_LM",
# ;     )

# ;     # ── Training args ─────────────────────────────────────────────────────────
# ;     # Ray Train sets LOCAL_RANK automatically; derive output paths from it
# ;     local_rank  = train.get_context().get_local_rank()
# ;     world_rank  = train.get_context().get_world_rank()
# ;     world_size = train.get_context().get_world_size()

# ;     train_dataset = config["train_dataset"].shard(num_shards=world_size, index=world_rank)

# ;     output_dir  = config.get("output_dir", "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-ray-z2")
# ;     save_dir    = config.get("save_dir",   "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-out-ray-z2")

# ;     training_args = SFTConfig(
# ;         output_dir=output_dir,
# ;         per_device_train_batch_size=2,
# ;         gradient_accumulation_steps=4,
# ;         optim = "adamw_bnb_8bit",   # The saviour _/\_
# ;         warmup_steps=5,
# ;         max_steps=50,
# ;         learning_rate=2e-4,
# ;         bf16=True,
# ;         save_strategy="steps",    
# ;         save_steps=10,   
# ;         save_total_limit=2,
# ;         gradient_checkpointing=True,
# ;         deepspeed = deepspeed_config,
# ;         report_to="wandb",
# ;         run_name="mistral-sft-ray",
# ;         max_length=2048,
# ;         dataset_text_field="text",
# ;         use_liger_kernel=True,
# ;         include_num_input_tokens_seen=True,
# ;     )

# ;     # ── W&B — init only on rank-0 ─────────────────────────────────────────────
# ;     if world_rank == 0:
# ;         wandb.init(
# ;             project="mistral_sft",
# ;             name="30steps-ray-z2",
# ;             dir="/home/yaswanthreddy/snorkel/monitoring",
# ;         )

# ;     # ── Trainer ───────────────────────────────────────────────────────────────
# ;     callbacks = (
# ;         [ThroughputMonitorCallback("/home/yaswanthreddy/snorkel/monitoring/mistral-sft-ray-z2.json")]
# ;         if world_rank == 0 else []
# ;     )

# ;     trainer = SFTTrainer(
# ;         model=model,
# ;         # peft_config=peft_config,   FUll sft for now, can add lora back later
# ;         processing_class=tokenizer,
# ;         train_dataset=train_dataset,
# ;         args=training_args,
# ;         callbacks=callbacks,
# ;     )

# ;     start = time.time()
# ;     trainer.train()
# ;     trainer.save_model(save_dir)

# ;     if world_rank == 0:
# ;         wandb.finish()
# ;         print(f"Total time: {time.time() - start:.1f} seconds")

# ;     # Report final loss back to the Ray driver
# ;     metrics = {k: v for k, v in trainer.state.log_history[-1].items()}
# ;     train.report(metrics)


# ; def train_fun(
# ;     model_path: str = "/data01/yaswanthreddy/models/mistral-7b-v0.1",
# ;     data_path:  str = "/home/yaswanthreddy/snorkel/data/ultrachat_200k_train_sft.jsonl",
# ;     num_workers: int = 2,
# ;     use_gpu:     bool = True,
# ; ):
# ;     """Ray driver — launches distributed training across `num_workers` GPUs."""

# ;     tokenizer = AutoTokenizer.from_pretrained(model_path)
# ;     tokenizer.pad_token     = tokenizer.eos_token
# ;     tokenizer.padding_side  = "right"
# ;     tokenizer.chat_template = (
# ;         "{{ bos_token }}"
# ;         "{% for message in messages %}"
# ;         "{% if message['role'] == 'user' %}"
# ;         "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
# ;         "{% elif message['role'] == 'assistant' %}"
# ;         "{{ message['content'] + eos_token }}"
# ;         "{% endif %}"
# ;         "{% endfor %}"
# ;     )

# ;     dataset = load_dataset("json", data_files=data_path, split="train")
# ;     dataset = dataset.remove_columns(["prompt", "prompt_id"])
# ;     split   = dataset.train_test_split(test_size=0.05)

# ;     original_cols = split["train"].column_names
# ;     train_dataset = split["train"].map(
# ;         apply_chat_template,
# ;         fn_kwargs={"tokenizer": tokenizer},
# ;         remove_columns=original_cols,
# ;     )

# ;     ray.init(ignore_reinit_error=True)

# ;     trainer = TorchTrainer(
# ;         train_loop_per_worker=train_loop_per_worker,
# ;         train_loop_config={
# ;             "model_path": model_path,
# ;             "data_path":  data_path,
# ;             "output_dir": "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-ray-z2",
# ;             "save_dir":   "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-out-ray-z2",
# ;             "deepspeed_config": "/home/yaswanthreddy/snorkel/configs/ds_zero2.json",
# ;             "train_dataset": train_dataset,
# ;         },
# ;         torch_config=TorchConfig(backend="nccl"), 
# ;         scaling_config=ScalingConfig(
# ;             num_workers=num_workers,
# ;             use_gpu=use_gpu,
# ;             resources_per_worker={"GPU": 1},
# ;         ),
# ;         run_config=RunConfig(
# ;             name="mistral-sft-ray",
# ;             checkpoint_config=CheckpointConfig(num_to_keep=2),
# ;             storage_path="/home/yaswanthreddy/snorkel/ray_results",
# ;         ),
# ;     )

# ;     result = trainer.fit()
# ;     print("Training result:", result.metrics)
# ;     return result


# ; if __name__ == "__main__":
# ;     train_fun()

