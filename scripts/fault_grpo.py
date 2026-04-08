import os
import glob
import json
import ray
import ray.train as train
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch, time, math, logging, re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
import wandb

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RANK%(message)s')
logger = logging.getLogger("grpo_fault_tolerant")

# ─── 1. Reward Functions (GRPO Logic) ─────────────────────────────────────────
def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion: score += 0.2
        if "</think>" in completion: score += 0.2
        if "<answer>" in completion: score += 0.2
        if "</answer>" in completion: score += 0.2
        pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
        if re.search(pattern, completion, re.DOTALL | re.IGNORECASE):
            score += 0.2
        rewards.append(score)
    return rewards

def correctness_reward_func(prompts, completions, **kwargs):
    """
    Standardized signature: (prompts, completions, **kwargs)
    We pull 'answer' (the ground truth) from kwargs.
    """
    # 🚨 CRITICAL: Pull the ground truth list from kwargs
    # In GRPO, this will be a list of truth-strings, one for each completion.
    truth_labels = kwargs.get("answer")
    
    if not truth_labels:
        return [0.0] * len(completions)

    rewards = []
    for comp, truth in zip(completions, truth_labels):
        match = re.search(r"<answer>(.*?)</answer>", comp, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted == str(truth).strip():
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# ─── 2. Full Observability FaultToleranceCallback ────────────────────────────
class FaultToleranceCallback(TrainerCallback):
    def __init__(self, max_consecutive_nan=3, grad_norm_threshold=100.0, 
                 memory_warn_pct=0.92, log_path=None):
        self.max_consecutive_nan = max_consecutive_nan
        self.grad_norm_threshold = grad_norm_threshold
        self.memory_warn_pct = memory_warn_pct
        self.log_path = log_path

        self.consecutive_nan_count = 0
        self.total_nan_count = 0
        self.total_grad_explosions = 0
        self.fault_events = []

    def _log_fault(self, step, fault_type, details=""):
        event = {
            "step": step,
            "type": fault_type,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        }
        self.fault_events.append(event)
        logger.warning(f"[Step {step}] FAULT: {fault_type} — {details}")

        if wandb.run is not None:
            wandb.log({
                f"faults/{fault_type}": 1,
                "faults/total_nan": self.total_nan_count,
                "faults/consecutive_nan": self.consecutive_nan_count,
            }, step=step)

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_pct = allocated / total
            if usage_pct > self.memory_warn_pct:
                self._log_fault(state.global_step, "memory_pressure", f"{usage_pct:.1%} usage")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if math.isnan(loss) or math.isinf(loss):
                self.consecutive_nan_count += 1
                self.total_nan_count += 1
                self._log_fault(state.global_step, "nan_loss", f"Loss={loss}")
                if self.consecutive_nan_count >= self.max_consecutive_nan:
                    logger.critical("Consecutive NaNs detected. Triggering rollback.")
                    control.should_training_stop = True
            else:
                self.consecutive_nan_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            grad_norm = state.log_history[-1].get("grad_norm", None)
            if grad_norm and grad_norm > self.grad_norm_threshold:
                self.total_grad_explosions += 1
                self._log_fault(state.global_step, "gradient_explosion", f"norm={grad_norm:.2f}")

    def on_train_end(self, args, state, control, **kwargs):
        report = {"events": self.fault_events, "nan_total": self.total_nan_count}
        if self.log_path:
            with open(self.log_path, "w") as f:
                json.dump(report, f, indent=2)
        if wandb.run is not None:
            wandb.log({"faults/summary_report": len(self.fault_events)})

# ─── 3. Checkpoint Discovery ──────────────────────────────────────────────────
def find_latest_checkpoint(output_dir: str):
    if not os.path.exists(output_dir): return None
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs: return None
    
    def extract_step(path):
        try: return int(path.split("-")[-1])
        except: return -1

    # Sort so we check newest first
    checkpoint_dirs.sort(key=extract_step, reverse=True)
    for latest in checkpoint_dirs:
        # Check for the 'trainer_state' to ensure the save actually finished
        if os.path.exists(os.path.join(latest, "trainer_state.json")):
            return latest
    return None

# ─── 4. Main Worker Loop with OOM & NCCL Recovery ────────────────────────────
def train_loop_per_worker(config: dict):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    world_rank = train.get_context().get_world_rank()
    output_dir = config["output_dir"]

    resume_checkpoint = find_latest_checkpoint(output_dir)

    # ── Model & Tokenizer ──
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset (GSM8K) ──
    raw_dataset = load_dataset("gsm8k", "main", split="train[:1000]")
    train_dataset = raw_dataset.map(lambda x: {
        "prompt": f"<s>[INST] Think in <think>. Answer in <answer>.\n\n{x['question']} [/INST]",
        "answer": x['answer'].split("#### ")[-1]
    })

    # Explicitly keep only what the trainer needs
    train_dataset = train_dataset.select_columns(["prompt", "answer"])

    world_size = train.get_context().get_world_size()
    world_rank = train.get_context().get_world_rank()

    train_dataset = train_dataset.shard(num_shards=world_size, index=world_rank)

    # ── OOM Flag Check (Memory reduction on restart) ──
    oom_flag_path = f"/tmp/grpo_oom_{world_rank}.flag"
    num_gens = config.get("num_generations", 8)
    if os.path.exists(oom_flag_path):
        num_gens = max(2, num_gens // 2)
        logger.warning(f"OOM Flag detected! Reducing num_generations to {num_gens}")

    # ── GRPO Config ──
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        optim = "adamw_bnb_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=config["max_steps"],
        bf16=True,
        num_generations=num_gens,
        generation_batch_size=num_gens,
        max_completion_length=2048,
        # deepspeed = config["deepspeed_config"],
        save_total_limit=3, # Restored limit
        report_to="wandb",
        use_liger_kernel=True,
        gradient_checkpointing=True,
        logging_steps=1,
        beta=0.04, 
    )

    if world_rank == 0:
        wandb.init(project="mistral_grpo_fault_tol", id=config["wandb_run_id"], resume="allow")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[format_reward_func, correctness_reward_func],
        # peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        processing_class=tokenizer,
        callbacks=[FaultToleranceCallback(log_path=f"/tmp/faults_rank{world_rank}.json")]
    )

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        if world_rank == 0:
            logger.info("Training complete. Saving final model...")
            trainer.save_model(os.path.join(output_dir, "final_model"))
        if world_rank == 0 and os.path.exists(oom_flag_path):
            os.remove(oom_flag_path)
    except torch.cuda.OutOfMemoryError:
        logger.error(f"RANK {world_rank} OOM! Setting flag and triggering restart.")
        torch.cuda.empty_cache()
        with open(oom_flag_path, "w") as f: f.write("1")
        raise RuntimeError("OOM - Watchdog Restart")
    except RuntimeError as e:
        if "NCCL" in str(e):
            logger.critical("NCCL Error detected. Multi-GPU sync failure. Restarting all ranks.")
            raise
        raise e
# reward_funcs=[lambda completions, **kwargs: [1.0 for _ in completions]], # Dummy reward
# ─── 5. Ray Driver ───────────────────────────────────────────────────────────
def launch_grpo():
    ray.init(ignore_reinit_error=True)

    model_id = "mistralai/Mistral-7B-v0.1" 
    lustre_path = "/lustre/nvwulf/scratch/ybhuma"
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model_path": model_id,
            "output_dir": f"{lustre_path}/checkpoints/grpo_mistral",
            "max_steps": 1500,
            "wandb_run_id": f"grpo-fault-tol-{int(time.time())}",
            "num_generations": 8,
            # "deepspeed_config": "configs/ds_zero2.json"
        },
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        run_config=RunConfig(
            failure_config=ray.train.FailureConfig(max_failures=3),
            checkpoint_config=CheckpointConfig(num_to_keep=3)
        )
    )
    trainer.fit()

if __name__ == "__main__":
    launch_grpo()