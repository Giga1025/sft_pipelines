import os
import glob
import json
import ray
import ray.train as train
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from datasets import load_dataset
import torch, time, math, logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
import wandb

from callbacker import ThroughputMonitorCallback
from chaos_monkey import ChaosInjectorCallback

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RANK%(message)s',
)
logger = logging.getLogger("fault_tolerant_trainer")


# ─── Fault Tolerance Callback ────────────────────────────────────────────────
# This is the INNER layer of fault tolerance.
# It hooks into HuggingFace Trainer's training loop at each step
# to detect and handle recoverable errors WITHOUT killing the process.

class FaultToleranceCallback(TrainerCallback):
    """
    Hooks into the HF Trainer loop to catch problems per-step:
      - NaN / Inf loss detection
      - Gradient norm explosion detection
      - GPU memory pressure warnings
    
    On recoverable failures: skips the step, zeros gradients, logs the event.
    On repeated failures (consecutive NaNs): signals the trainer to stop
    so the outer watchdog can restart from checkpoint.
    """

    def __init__(self, max_consecutive_nan=3, grad_norm_threshold=100.0, 
                 memory_warn_pct=0.92, log_path=None):
        self.max_consecutive_nan = max_consecutive_nan
        self.grad_norm_threshold = grad_norm_threshold
        self.memory_warn_pct = memory_warn_pct
        self.log_path = log_path

        # Tracking state
        self.consecutive_nan_count = 0
        self.total_nan_count = 0
        self.total_oom_skips = 0
        self.total_grad_explosions = 0
        self.fault_events = []  # log of every fault event

    def _log_fault(self, step, fault_type, details=""):
        """Record a fault event for post-mortem analysis."""
        event = {
            "step": step,
            "type": fault_type,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }
        self.fault_events.append(event)
        logger.warning(f"[Step {step}] FAULT: {fault_type} — {details}")

        # Also log to W&B if it's active
        if wandb.run is not None:
            wandb.log({
                f"faults/{fault_type}": 1,
                "faults/total_nan": self.total_nan_count,
                "faults/total_oom_skips": self.total_oom_skips,
                "faults/consecutive_nan": self.consecutive_nan_count,
            }, step=step)

    def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return

            # 🚨 FIX: Only evaluate if the log dictionary actually contains a step 'loss'
            if "loss" in logs:
                current_loss = logs["loss"]
                
                if math.isnan(current_loss) or math.isinf(current_loss):
                    self.consecutive_nan_count += 1
                    self.total_nan_count += 1
                    self._log_fault(state.global_step, "nan_loss", 
                                f"Loss={current_loss}, consecutive={self.consecutive_nan_count}")

                    if self.consecutive_nan_count >= self.max_consecutive_nan:
                        logger.critical(
                            f"[Step {state.global_step}] {self.consecutive_nan_count} consecutive NaN losses. "
                            f"Stopping training for checkpoint rollback."
                        )
                        control.should_training_stop = True
                else:
                    # Only reset the counter if we see a VALID loss number
                    self.consecutive_nan_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Before each training step: check GPU memory pressure.
        
        If memory is critically high, we log a warning. 
        This gives you data to correlate memory pressure with OOM events.
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_pct = allocated / total

            if usage_pct > self.memory_warn_pct:
                self._log_fault(
                    state.global_step, "memory_pressure",
                    f"GPU memory at {usage_pct:.1%} ({allocated/1e9:.1f}GB / {total/1e9:.1f}GB)"
                )

    def on_step_end(self, args, state, control, **kwargs):
        """
        After each training step: check gradient norms if available.
        """
        # HF Trainer logs grad_norm if max_grad_norm is set
        if state.log_history:
            last_log = state.log_history[-1]
            grad_norm = last_log.get("grad_norm", None)
            
            if grad_norm is not None and grad_norm > self.grad_norm_threshold:
                self.total_grad_explosions += 1
                self._log_fault(
                    state.global_step, "gradient_explosion",
                    f"grad_norm={grad_norm:.2f} (threshold={self.grad_norm_threshold})"
                )

    def on_train_end(self, args, state, control, **kwargs):
        """Save fault report at the end of training."""
        report = {
            "total_nan_losses": self.total_nan_count,
            "total_oom_skips": self.total_oom_skips,
            "total_gradient_explosions": self.total_grad_explosions,
            "total_fault_events": len(self.fault_events),
            "events": self.fault_events,
        }

        if self.log_path:
            with open(self.log_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Fault report saved to {self.log_path}")

        if wandb.run is not None:
            wandb.log({"faults/summary": report})


# ─── Checkpoint Discovery ────────────────────────────────────────────────────
# When the process restarts (via outer watchdog), we need to find the latest
# valid checkpoint and resume from it.

def find_latest_checkpoint(output_dir: str):
    """
    Scans the output directory for HuggingFace-style checkpoints 
    (checkpoint-{step}/) and returns the path to the most recent one.
    
    Returns None if no valid checkpoint exists (i.e., fresh start).
    
    WHY this function matters:
    When the outer watchdog restarts the process after a crash, this is 
    how the training script knows where to pick up. Without this, every 
    restart would train from scratch — wasting all the compute before the crash.
    """
    if not os.path.exists(output_dir):
        return None

    # HF Trainer saves checkpoints as checkpoint-{global_step}/
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if not checkpoint_dirs:
        return None

    # Sort by step number to find the latest
    def extract_step(path):
        try:
            return int(path.split("checkpoint-")[-1])
        except ValueError:
            return -1

    checkpoint_dirs = [d for d in checkpoint_dirs if extract_step(d) >= 0]
    
    if not checkpoint_dirs:
        return None

    latest = max(checkpoint_dirs, key=extract_step)

    # Basic integrity check: does it contain the essential files?
    # For DeepSpeed ZeRO-2/3, check for the deepspeed directory or adapter_config for LoRA
    has_adapter = os.path.exists(os.path.join(latest, "adapter_config.json"))
    has_ds_state = len(glob.glob(os.path.join(latest, "global_step*"))) > 0
    has_trainer_state = os.path.exists(os.path.join(latest, "trainer_state.json"))

    if (has_adapter or has_ds_state) and has_trainer_state:
        step = extract_step(latest)
        logger.info(f"Found valid checkpoint at {latest} (step {step})")
        return latest
    else:
        logger.warning(f"Checkpoint {latest} appears corrupted. Skipping.")
        checkpoint_dirs.remove(latest)
        if checkpoint_dirs:
            fallback = max(checkpoint_dirs, key=extract_step)
            logger.info(f"Falling back to {fallback}")
            return fallback
        return None


# ─── Main Training Function ──────────────────────────────────────────────────

def train_loop_per_worker(config: dict):
    """Ray Train worker function — runs on each GPU worker."""

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["NCCL_IB_DISABLE"] = "1"

    world_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()

    model_path = config["model_path"]
    data_path  = config["data_path"]
    output_dir = config.get("output_dir", "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-ray-z2")
    save_dir   = config.get("save_dir",   "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-out-ray-z2")

    # ── Check for existing checkpoint (RESUME LOGIC) ─────────────────────────
    # This is the bridge between the outer watchdog and inner recovery.
    # If we crashed and got restarted, this finds where we left off.
    resume_checkpoint = find_latest_checkpoint(output_dir)
    if resume_checkpoint:
        logger.info(f"[Rank {world_rank}] RESUMING from checkpoint: {resume_checkpoint}")
    else:
        logger.info(f"[Rank {world_rank}] No checkpoint found — starting fresh.")

    # ── Model & tokenizer ────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.select(range(1000))
    dataset = dataset.shard(num_shards=world_size, index=world_rank)
    dataset = dataset.remove_columns(["prompt", "prompt_id"])
    split   = dataset.train_test_split(test_size=0.05)

    original_cols = split["train"].column_names
    train_dataset = split["train"].map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=original_cols,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        optim="adamw_torch",
        warmup_steps=5,
        max_steps=40,
        learning_rate=2e-4,
        bf16=True,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,          # Keep 3 checkpoints for rollback safety
        gradient_checkpointing=True,
        deepspeed=config["deepspeed_config"],
        report_to="wandb",
        run_name="mistral-sft-ray",
        max_length=2048,
        dataset_text_field="text",
        use_liger_kernel=True,
        include_num_input_tokens_seen=True,
        max_grad_norm=1.0,           # Gradient clipping — catches explosions
        logging_steps=1,             # Log every step so NaN detection works promptly
    )

    oom_flag_path = "/tmp/oom_reduce_bs.flag"
    if os.path.exists(oom_flag_path):
        old_bs = training_args.per_device_train_batch_size
        new_bs = max(1, old_bs // 2)
        training_args.per_device_train_batch_size = new_bs
        logger.warning(f"[Rank {world_rank}] 🚨 OOM Flag detected! Reduced batch size: {old_bs} → {new_bs}")

    # ── W&B — resume-aware init on rank-0 only ────────────────────────────────
    # If resuming, we want to continue the SAME W&B run, not create a new one.
    # This gives you one continuous loss curve even across crashes.
    if world_rank == 0:
            wandb_id = config.get("wandb_run_id", None)

            wandb.init(
                project="mistral_sft",
                name="30steps-ray-z2",
                id=wandb_id,
                resume="allow",           # 🚨 ALWAYS set to "allow"
                dir="/home/yaswanthreddy/snorkel/monitoring",
            )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    fault_callback = FaultToleranceCallback(
        max_consecutive_nan=3,
        grad_norm_threshold=100.0,
        memory_warn_pct=0.92,
        log_path=(
            f"/home/yaswanthreddy/snorkel/monitoring/fault_report_rank{world_rank}.json"
            if world_rank == 0 else None
        ),
    )

    # chaos_callback = ChaosInjectorCallback(oom_step=5, nan_step=15, crash_step=25)
    
    # Chaos monkey MUST be first in the list!
    # Add chaos_callback for testing fault tolerance before fault_callback, but be careful — it will cause actual crashes and OOMs!
    callbacks = [fault_callback]

    if world_rank == 0:
        callbacks.append(
            ThroughputMonitorCallback(
                "/home/yaswanthreddy/snorkel/monitoring/mistral-sft-ray-z2.json"
            )
        )


    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    # ── Training with OOM protection ─────────────────────────────────────────
    # The outer try/except catches hard OOM that kills the whole train() call.
    # The FaultToleranceCallback handles per-step issues inside the loop.

    start = time.time()

    try:
        # trainer.train() handles resuming automatically if resume_checkpoint is passed
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        # 🚨 NEW: Did the trainer exit early because of NaNs?
        if fault_callback.consecutive_nan_count >= fault_callback.max_consecutive_nan:
            logger.error(f"[Rank {world_rank}] Training aborted due to NaN loss! Triggering Ray restart.")
            raise RuntimeError("NaN Loss - Triggering Ray Watchdog Restart")

        # If training succeeds completely, clean up the OOM flag so future runs start fresh
        if world_rank == 0 and os.path.exists(oom_flag_path):
            os.remove(oom_flag_path)

    except torch.cuda.OutOfMemoryError:
        logger.error(f"[Rank {world_rank}] OOM caught! In multi-GPU training, we must restart all ranks.")
        torch.cuda.empty_cache()
        
        # Drop the flag so the next Ray retry knows to lower the batch size
        with open(oom_flag_path, "w") as f:
            f.write("1")
            
        # Raise an error to intentionally kill the worker and trigger Ray's FailureConfig
        raise RuntimeError("OOM - Triggering Ray Watchdog Restart")

    except RuntimeError as e:
        if "NCCL" in str(e):
            logger.critical(f"[Rank {world_rank}] NCCL error: {e}")
            logger.critical("NCCL failures require full process restart. Exiting for watchdog.")
            raise  # Watchdog handles this
        else:
            raise  # Unknown error — don't swallow it

    # ── Save final model ──────────────────────────────────────────────────────
    if fault_callback.consecutive_nan_count == 0:
        trainer.save_model(save_dir)

    if world_rank == 0:
        elapsed = time.time() - start
        logger.info(f"Model saved to {save_dir}")
        logger.info(f"Total time: {elapsed:.1f}s")

        # Log fault summary to W&B
        if fault_callback.fault_events:
            wandb.log({
                "final/total_faults": len(fault_callback.fault_events),
                "final/nan_losses": fault_callback.total_nan_count,
                # "final/oom_retries": oom_retry_count,
            })
        wandb.finish()

    # Report final metrics to Ray driver
    metrics = {k: v for k, v in trainer.state.log_history[-1].items()}
    metrics["fault_events"] = len(fault_callback.fault_events)
    train.report(metrics)


# ─── Ray Driver ───────────────────────────────────────────────────────────────

def train_fun(
    model_path:  str = "/data01/yaswanthreddy/models/mistral-7b-v0.1",
    data_path:   str = "/home/yaswanthreddy/snorkel/data/ultrachat_200k_train_sft.jsonl",
    num_workers: int = 1,
    use_gpu:     bool = True,
    wandb_run_id: str = None,   # Pass a fixed ID to enable W&B resume across crashes
):
    """Ray driver — launches distributed training across `num_workers` GPUs."""

    ray.init(ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model_path": model_path,
            "data_path":  data_path,
            "output_dir": "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-ray-z2",
            "save_dir":   "/home/yaswanthreddy/snorkel/checkpoints/mistral-sft-out-ray-z2",
            "deepspeed_config": "/home/yaswanthreddy/snorkel/configs/ds_zero2.json",
            "wandb_run_id": wandb_run_id or f"mistral-sft-{int(time.time())}",
        },
        torch_config=TorchConfig(backend="nccl"),
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={"GPU": 1},
        ),
        run_config=RunConfig(
            name="mistral-sft-ray",
            checkpoint_config=CheckpointConfig(num_to_keep=3),
            storage_path="/home/yaswanthreddy/snorkel/ray_results",
            # Ray's own failure handling — complements the inner layer
            failure_config=ray.train.FailureConfig(max_failures=3),
        ),
    )

    result = trainer.fit()
    print("Training result:", result.metrics)
    return result


if __name__ == "__main__":
    train_fun()