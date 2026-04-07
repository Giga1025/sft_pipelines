import torch
import time
import os
import json
from transformers import TrainerCallback

class ThroughputMonitorCallback(TrainerCallback):
    def __init__(self, log_file="throughput_log.json"):
        self.log_file = log_file
        self.step_times = []
        self.step_start = None
        self.results = {"config": {}, "steps": [], "summary": {}}

        # --- Trackers for logging_steps > 1 ---
        self._prev_num_tokens = 0
        self._last_log_len = 0

    def on_train_begin(self, args, state, control, **kwargs):
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
        
        self.results["config"] = {
            "parallelism":            os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "per_device_batch_size":  args.per_device_train_batch_size,
            "gradient_accum_steps":   args.gradient_accumulation_steps,
            "effective_batch_size":   args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size,
            "bf16":                   args.bf16,
            "fp16":                   args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "liger_kernel":           getattr(args, "use_liger_kernel", False),
            "max_steps":              args.max_steps,
            "num_gpus":               world_size,
        }
        
        self.train_start = time.time()
        self.total_tokens = 0
        self._prev_num_tokens = 0
        self._last_log_len = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - self.step_start

        # --- Memory Stats ---
        allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        reserved  = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        peak      = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        try:
            gpu_util = torch.cuda.utilization()
        except (NotImplementedError, RuntimeError, AttributeError):
            gpu_util = 0

        # --- EXACT TOKEN MATH (Works natively regardless of logging_steps) ---
        cumulative_tokens = getattr(state, "num_input_tokens_seen", 0)
        
        if cumulative_tokens > self._prev_num_tokens:
            tokens_this_step = cumulative_tokens - self._prev_num_tokens
            self._prev_num_tokens = cumulative_tokens
        else:
            # Fallback estimate (includes world_size multiplier)
            seq_len = getattr(args, "max_seq_length", None) or getattr(args, "max_length", 2048)
            world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
            tokens_this_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * seq_len * world_size

        self.total_tokens += tokens_this_step
        tokens_per_sec = tokens_this_step / step_time if step_time > 0 else 0

        # --- LOSS LOGIC (Only grabs fresh loss on logging steps) ---
        loss, grad_norm = None, None
        if state.log_history and len(state.log_history) > self._last_log_len:
            latest = state.log_history[-1]
            loss = latest.get("loss")
            grad_norm = latest.get("grad_norm")
            self._last_log_len = len(state.log_history)

        step_data = {
            "step":             state.global_step,
            "step_time_sec":    round(step_time, 3),
            "tokens_per_sec":   round(tokens_per_sec, 1),
            "gpu_util_pct":     gpu_util,
            "vram_alloc_gb":    round(allocated, 2),
            "vram_reserved_gb": round(reserved, 2),
            "vram_peak_gb":     round(peak, 2),
            "loss":             loss,
            "grad_norm":        grad_norm,
        }

        self.step_times.append(step_time)
        self.results["steps"].append(step_data)

        # Dynamic string building for clean terminal output
        loss_str = f" | Loss: {loss:.4f}" if loss is not None else ""
        gpu_str = f" | GPU: {gpu_util}%" if gpu_util > 0 else ""
        
        print(
            f"Step {state.global_step:>3} | "
            f"Time: {step_time:.2f}s | "
            f"Tok/s: {tokens_per_sec:,.0f}"
            f"{gpu_str} | "
            f"VRAM: {allocated:.1f}/{reserved:.1f} GB"
            f"{loss_str}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start

        # --- Guard against short runs (skip up to 10 warmup steps) ---
        warmup_skip = min(10, max(0, len(self.step_times) - 1))
        stable_steps = self.step_times[warmup_skip:] if warmup_skip > 0 else self.step_times
        avg_step = sum(stable_steps) / len(stable_steps) if stable_steps else 0.0

        # Recompute tokens/sec from stable window only
        if len(self.results["steps"]) > warmup_skip:
            stable_token_data = [
                s["tokens_per_sec"] * s["step_time_sec"] 
                for s in self.results["steps"][warmup_skip:]
            ]
            stable_total_tokens = sum(stable_token_data)
            stable_total_time = sum(stable_steps)
            stable_avg_tps = round(stable_total_tokens / stable_total_time, 1) if stable_total_time > 0 else 0.0
        else:
            stable_avg_tps = 0.0

        self.results["summary"] = {
            "total_time_sec":       round(total_time, 1),
            "warmup_steps_skipped": warmup_skip,
            "stable_steps_used":    len(stable_steps),
            "avg_step_time_sec":    round(avg_step, 3),
            "avg_tokens_per_sec":   stable_avg_tps,
            "peak_vram_gb":         round((torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0) / 1e9, 2),
            "total_tokens_trained": self.total_tokens,
        }

        print("\n===== THROUGHPUT SUMMARY =====")
        for k, v in self.results["summary"].items():
            print(f"  {k}: {v}")

        with open(self.log_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved to {self.log_file}")