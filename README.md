# sft_pipelines

A fault-tolerant training harness for SFT and GRPO on top of HuggingFace TRL,
Ray Train, and DeepSpeed. Built to actually understand fault tolerance in
distributed LLM training , by making training break on purpose and watching it
recover.

The harness has been used across Mistral-7B, Phi-4 (14B), and Qwen-2.5 (3B / 7B / 14B / 32B),
on hardware ranging from a single A6000 to multi-node H100/H200.

---

## Why this exists

Reading about fault tolerance is not the same as building it. This repo is what
came out of trying to *actually* understand what happens when a distributed
training run goes wrong , OOMs mid-step, NaN losses, exploding gradients, NCCL
crashes , and how to recover from those without a human at the keyboard.

The interesting parts are not the training scripts themselves. They are the
two-layer recovery design and the chaos monkey that verifies it works.

---

## What's in here

### Two-layer fault tolerance

**Inner layer , `FaultToleranceCallback`** (`scripts/fault_tol.py`)

Hooks HuggingFace `Trainer` per step. Catches recoverable problems without
killing the process:

- **NaN / Inf loss detection.** Skips the offending step, zeros gradients,
  increments a consecutive-NaN counter. After N consecutive NaNs the callback
  signals `should_training_stop = True` so the outer watchdog can restart from
  the last good checkpoint.
- **Gradient-norm explosion detection.** Reads `grad_norm` from the trainer's
  log history and warns / records the event when it exceeds a configurable
  threshold.
- **GPU memory pressure warnings.** Checks `torch.cuda.memory_allocated()` vs.
  total VRAM before each step and records a `memory_pressure` event when usage
  crosses ~92%.

Every fault event is logged with step, type, timestamp, and GPU memory state,
and mirrored to Weights & Biases if a W&B run is active.

**Outer layer , Ray watchdog + checkpoint resume**

The training loop runs inside a Ray `TorchTrainer` (`ScalingConfig`,
`RunConfig`, `CheckpointConfig`). On a hard crash , for example the injected
"NCCL" error from the chaos monkey , Ray restarts the worker, and
`find_latest_checkpoint()` scans the output directory for HuggingFace-style
`checkpoint-{step}/` folders and resumes from the most recent valid one, with
a fallback to the next-latest if the newest is corrupt.

Together this gives per-step recovery for soft failures and full checkpoint
resume for hard failures , no manual intervention.

### Chaos monkey

**`ChaosInjectorCallback`** (`scripts/chaos_monkey.py`)

Deliberately breaks the training loop at configurable steps to verify the
recovery layers actually work:

- **OOM injection** , allocates a giant CUDA tensor at a target step to trip
  the memory-pressure path.
- **NCCL / hard-crash injection** , raises a `RuntimeError` masquerading as a
  collective-communication failure so the outer Ray watchdog has to restart
  and resume.
- **NaN injection hook** , the callback also supports scheduling NaN faults
  (paired with the inner `FaultToleranceCallback`'s NaN-loss handling).

File-based flags in `/tmp/chaos_flags/` ensure each fault fires exactly once
across restarts, so you can watch a run recover through multiple different
failure modes end-to-end without an infinite loop.

### GRPO with reward functions

`scripts/fault_grpo.py` runs GRPO under the same two-layer fault tolerance,
with two reward functions:

- `format_reward_func` , rewards `<think>...</think><answer>...</answer>`
  structure.
- `correctness_reward_func` , compares generated answers against ground truth
  pulled from `kwargs`.

### Training configs

`configs/` , DDP, DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3 configs (bf16, gradient
clipping, overlap comm, contiguous gradients, reduce-scatter). Set up so the
same training code can be run under different parallelism strategies without
changing the code, only the config.

### Throughput monitoring

`scripts/callbacker.py` , `ThroughputMonitorCallback` records per-step time,
tokens/sec, allocated/reserved GPU memory, and effective batch size. Useful
for comparing ZeRO stages / DDP / gradient checkpointing settings against each
other on the same hardware.

### Data versioning

`.dvc/` and `data/ultrachat_200k_train_sft.jsonl.dvc` , dataset tracked with
DVC so training runs are reproducible.

---

## Repo layout

```
sft_pipelines/
├── scripts/
│   ├── fault_tol.py       # SFT training loop with inner+outer fault tolerance
│   ├── fault_grpo.py      # GRPO training loop with the same recovery layers
│   ├── chaos_monkey.py    # Deliberate OOM / NaN / NCCL fault injection
│   ├── callbacker.py      # Throughput and memory monitoring
│   ├── ray_train.py       # Baseline Ray Train SFT loop
│   └── ...
├── configs/
│   ├── ddp.yaml
│   ├── deepspeed_zero1.yaml
│   ├── ds_zero1.json
│   ├── ds_zero2.json
│   └── ds_zero3.json
├── data/                  # DVC-tracked training data
└── README.md
```

---

## Stack

- PyTorch, HuggingFace Transformers, TRL (`SFTTrainer`, `GRPOTrainer`)
- Ray Train (`TorchTrainer`) for distributed orchestration and restart
- DeepSpeed (ZeRO 1/2/3) and DDP
- PEFT (LoRA)
- Weights & Biases for run tracking and fault-event logging
- DVC for dataset versioning

---

## Notes

The models and hardware span above are what the harness has actually been run
against , because the recovery layers hook `Trainer` through TRL, the same
callbacks drop in cleanly regardless of which model or backend is underneath.

This is a personal learning / practice repo, not a production framework.
