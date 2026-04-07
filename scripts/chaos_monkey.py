import torch
import logging
import os
from transformers import TrainerCallback

logger = logging.getLogger("fault_tolerant_trainer")

class ChaosInjectorCallback(TrainerCallback):
    """
    Intentionally injects critical failures to test the resilience of the training loop.
    Uses file-based flags to ensure failures are only injected ONCE across Ray restarts.
    """
    def __init__(self, oom_step=5, nan_step=15, crash_step=25):
        self.oom_step = oom_step
        self.nan_step = nan_step
        self.crash_step = crash_step
        self.flag_dir = "/tmp/chaos_flags"
        os.makedirs(self.flag_dir, exist_ok=True)

    def _should_inject(self, fault_name):
        """Check if we already injected this fault in a previous life."""
        flag_path = os.path.join(self.flag_dir, f"{fault_name}.flag")
        if os.path.exists(flag_path):
            return False
        # Drop the flag so we never inject this specific fault again
        open(flag_path, 'w').close()
        return True

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step

        # 1. Inject Out Of Memory (OOM)
        if step == self.oom_step and self._should_inject("oom"):
            logger.warning(f"🐒 [Chaos Monkey] Injecting Fake OOM at step {step}...")
            self.killer_tensor = torch.empty((100000, 100000, 1000), device="cuda")

        # 3. Inject Hard Crash 
        if step == self.crash_step and self._should_inject("crash"):
            logger.warning(f"🐒 [Chaos Monkey] Injecting Fake NCCL Crash at step {step}...")
            raise RuntimeError("Fake NCCL communication error injected for testing.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Hijack the logs to inject a true NaN loss for 3 consecutive steps."""
        step = state.global_step
        
        # 2. Inject NaN Loss (for exactly 3 steps starting at nan_step)
        if self.nan_step <= step < self.nan_step + 3:
            # We use a unique flag for each of the 3 steps so it doesn't skip the consecutive check on restart
            if self._should_inject(f"nan_log_{step}"):
                if logs is not None:
                    logs['loss'] = float('nan')
                    logger.warning(f"🐒 [Chaos Monkey] Hijacked logs to inject NaN at step {step}...")