"""
train.py — Step 6: Training Loop

Project        : VarianceEngine
Pipeline stage : Step 6 of 9 (see PIPELINE.md)

Usage
-----
    python src/train.py --config configs/train_config.yaml

    # Override specific values:
    python src/train.py --config configs/train_config.yaml \\
        training.batch_size=2 training.grad_accum_steps=8

Execution model
---------------
Single-GPU training on RTX 4090 (24 GB VRAM).
RTX 4080 (16 GB) can be used with batch_size=2, grad_accum_steps=8.

Mixed precision (bf16):
    RTX 4090 has native bf16 support. bf16 is more numerically stable than
    fp16 for transformer training (no loss scaling needed) and halves VRAM
    for activations and optimizer states vs fp32.

Gradient accumulation:
    effective_batch_size = batch_size × grad_accum_steps = 4 × 4 = 16.
    Simulates large batch training within VRAM constraints.

Gradient checkpointing:
    Recomputes activations during backward pass instead of storing them.
    Reduces VRAM by ~40% at ~30% compute cost. Required for batch_size=4
    with LoRA rank=32 on 24 GB VRAM.

Dual learning rates:
    conditioner (new layers):  lr=1e-3  — randomly initialised, needs fast convergence
    lora (adapted attention):  lr=1e-4  — pre-trained, conservative update

Checkpointing strategy:
    - Save every `save_every_steps` steps (latest-N rolling window)
    - Always save best val loss checkpoint separately
    - Checkpoint = LoRA weights + conditioner only (~50 MB)
    - Early stopping: stop if val loss doesn't improve for `patience` evaluations
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import build_dataloaders
from src.model.variance_engine import VarianceEngineModel
from src.model.lora_utils import print_parameter_summary


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    split_pairs_path: str = "outputs/split_pairs.json"
    preprocessed_dir: str = "preprocessed/"
    output_dir: str = "checkpoints/"
    log_dir: str = "logs/"

    # Model
    musicgen_name: str = "facebook/musicgen-medium"
    lora_rank: int = 32
    conditioner_dropout: float = 0.1

    # Audio
    sample_rate: int = 32_000
    max_duration_s: float = 15.0
    encodec_frame_rate: int = 50

    # Training
    n_epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    lr_conditioner: float = 1e-3
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    save_every_steps: int = 500
    keep_last_n_checkpoints: int = 3
    early_stopping_patience: int = 5
    val_every_steps: int = 200

    # Reproducibility
    seed: int = 42

    # Logging
    use_wandb: bool = True
    wandb_project: str = "VarianceEngine"
    wandb_entity: Optional[str] = None
    log_every_steps: int = 10


def load_config(config_path: str, overrides: list[str]) -> TrainConfig:
    """Load YAML config and apply dot-notation CLI overrides.

    Override format: 'section.key=value', e.g. 'training.batch_size=2'.
    Type is inferred from the YAML default.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Flatten nested YAML into TrainConfig fields
    flat = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values

    # Apply CLI overrides
    for override in overrides:
        key, _, val_str = override.partition("=")
        key = key.split(".")[-1]  # strip section prefix
        if key in flat:
            orig = flat[key]
            if isinstance(orig, bool):
                flat[key] = val_str.lower() in ("true", "1", "yes")
            elif isinstance(orig, int):
                flat[key] = int(val_str)
            elif isinstance(orig, float):
                flat[key] = float(val_str)
            else:
                flat[key] = val_str

    cfg = TrainConfig(**{k: v for k, v in flat.items() if k in TrainConfig.__dataclass_fields__})
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Rolling checkpoint window + best-val-loss checkpoint.

    Keeps the last `keep_n` step checkpoints and always preserves the
    checkpoint with the lowest validation loss.

    Rationale for keeping last-N rather than best-N:
        The best checkpoint by val loss can be retrieved separately. Keeping
        recent checkpoints allows resuming from a recent step if training is
        interrupted — useful when training on a server without persistent
        sessions.
    """

    def __init__(self, output_dir: Path, keep_n: int = 3):
        self.output_dir = output_dir
        self.keep_n = keep_n
        self.saved_steps: list[int] = []
        self.best_val_loss: float = float("inf")
        output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: VarianceEngineModel,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step: int,
        val_loss: Optional[float] = None,
    ) -> Path:
        """Save checkpoint and maintain rolling window.

        Returns path of saved checkpoint.
        """
        ckpt_path = self.output_dir / f"step_{step:07d}.pt"
        torch.save(
            {
                "step": step,
                "val_loss": val_loss,
                "conditioner_state": model.conditioner.state_dict(),
                "lora_state": {
                    k: v for k, v in model.transformer.state_dict().items()
                    if "lora_A" in k or "lora_B" in k
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            ckpt_path,
        )

        # Rolling window
        self.saved_steps.append(step)
        if len(self.saved_steps) > self.keep_n:
            old_step = self.saved_steps.pop(0)
            old_path = self.output_dir / f"step_{old_step:07d}.pt"
            if old_path.exists():
                old_path.unlink()

        # Best val loss
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.output_dir / "best.pt"
            shutil.copy2(ckpt_path, best_path)
            print(f"  New best val loss: {val_loss:.4f} → saved best.pt")

        return ckpt_path

    def load_latest(
        self,
        model: VarianceEngineModel,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
    ) -> int:
        """Resume from the latest checkpoint. Returns step number."""
        if not self.saved_steps:
            # Scan output dir for existing checkpoints
            existing = sorted(self.output_dir.glob("step_*.pt"))
            if not existing:
                return 0
            ckpt_path = existing[-1]
        else:
            ckpt_path = self.output_dir / f"step_{self.saved_steps[-1]:07d}.pt"

        ckpt = torch.load(ckpt_path, map_location=device)
        model.conditioner.load_state_dict(ckpt["conditioner_state"])
        model.transformer.load_state_dict(ckpt["lora_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        step = ckpt["step"]
        print(f"Resumed from {ckpt_path} (step {step})")
        return step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model: VarianceEngineModel,
    val_loader,
    device: str,
    use_bf16: bool,
) -> float:
    """Compute mean val loss over the full validation set.

    Returns
    -------
    float — mean cross-entropy loss over all valid token positions.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(val_loader, desc="  Val", leave=False):
        gt_audio = batch["gt_audio"].to(device)
        var_audio = batch["var_audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Encode variation to codes (fp32: EnCodec LSTM does not support bf16)
        with torch.cuda.amp.autocast(enabled=False):
            var_codes, _ = model.compression_model.encode(var_audio.float())

        with autocast(dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(gt_audio, var_codes, attention_mask)
            loss = VarianceEngineModel.compute_loss(logits, var_codes, attention_mask)

        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# LR schedule (pure PyTorch — avoids transformers version dependency)
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine decay with linear warmup, implemented via LambdaLR.

    Identical behaviour to transformers.get_cosine_schedule_with_warmup
    but has no transformers dependency.

    During warmup (steps 0 → num_warmup_steps): lr scales linearly 0 → base_lr.
    After warmup: lr follows cosine decay from base_lr → min_lr_ratio * base_lr.

    min_lr_ratio=0.0 means LR decays to zero at num_training_steps.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = cfg.precision == "bf16" and device == "cuda"
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print("=" * 60)
    print("VarianceEngine — Training — Step 6")
    print(f"Device: {device} | Precision: {cfg.precision}")
    print(f"Effective batch size: {cfg.batch_size} × {cfg.grad_accum_steps} = "
          f"{cfg.batch_size * cfg.grad_accum_steps}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=cfg.__dict__,
                name=f"lora_r{cfg.lora_rank}_lr{cfg.lr_lora}",
            )
        except ImportError:
            print("wandb not installed — logging to console only.")
            cfg.use_wandb = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nBuilding DataLoaders...")
    train_loader, val_loader, _ = build_dataloaders(
        split_pairs_path=cfg.split_pairs_path,
        sample_rate=cfg.sample_rate,
        max_duration_s=cfg.max_duration_s,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.n_epochs
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nInitialising model...")
    model = VarianceEngineModel(
        musicgen_name=cfg.musicgen_name,
        lora_rank=cfg.lora_rank,
        conditioner_dropout=cfg.conditioner_dropout,
        device=device,
    )

    if cfg.gradient_checkpointing:
        # Enable gradient checkpointing on the transformer backbone.
        # audiocraft's StreamingTransformer supports gradient checkpointing
        # via the standard HuggingFace interface if the underlying model
        # inherits from PreTrainedModel, or via torch.utils.checkpoint directly.
        # We enable it if the transformer exposes the method.
        if hasattr(model.transformer, "gradient_checkpointing_enable"):
            model.transformer.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled.")
        else:
            print("  Warning: transformer does not expose gradient_checkpointing_enable(). "
                  "Checkpointing not applied — VRAM may be higher than expected.")

    model.train()

    # ------------------------------------------------------------------
    # Optimizer — dual learning rates
    # ------------------------------------------------------------------
    # Two parameter groups: conditioner (new, lr=1e-3) and LoRA (lr=1e-4).
    # The ratio 10× between groups is intentional: the conditioner's projection
    # is randomly initialised and must converge faster than the LoRA adapters
    # which fine-tune pre-trained attention weights.
    conditioner_params = list(model.conditioner.parameters())
    lora_params = [p for p in model.transformer.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": conditioner_params, "lr": cfg.lr_conditioner, "name": "conditioner"},
            {"params": lora_params,        "lr": cfg.lr_lora,        "name": "lora"},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),   # β2=0.95 (vs default 0.999): faster adaptation
                              # for fine-tuning on small datasets; recommended
                              # in LoRA literature for convergence stability.
    )

    # ------------------------------------------------------------------
    # LR Schedule — cosine with linear warmup
    # ------------------------------------------------------------------
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Checkpoint manager
    # ------------------------------------------------------------------
    ckpt_manager = CheckpointManager(output_dir, keep_n=cfg.keep_last_n_checkpoints)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    global_step = 0          # optimizer steps (after grad accumulation)
    raw_step = 0             # forward pass steps (before accumulation)
    best_val_loss = float("inf")
    patience_counter = 0
    running_loss = 0.0

    optimizer.zero_grad()

    print("\nStarting training...\n")

    for epoch in range(cfg.n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}", unit="batch")

        for batch in pbar:
            gt_audio      = batch["gt_audio"].to(device)
            var_audio     = batch["var_audio"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # ----------------------------------------------------------
            # Encode variation to EnCodec discrete codes (frozen codec)
            # fp32 required: EnCodec LSTM does not support bf16
            # ----------------------------------------------------------
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    var_codes, _ = model.compression_model.encode(var_audio.float())
                # var_codes: (B, n_q, T_frames)

            # ----------------------------------------------------------
            # Forward + loss
            # ----------------------------------------------------------
            with autocast(dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(gt_audio, var_codes, attention_mask)
                loss = VarianceEngineModel.compute_loss(logits, var_codes, attention_mask)
                loss = loss / cfg.grad_accum_steps  # scale for accumulation

            loss.backward()

            running_loss += loss.item() * cfg.grad_accum_steps  # unscale for logging
            raw_step += 1

            # ----------------------------------------------------------
            # Optimizer step (after accumulating grad_accum_steps batches)
            # ----------------------------------------------------------
            if raw_step % cfg.grad_accum_steps == 0:
                # Gradient clipping before step
                # Clips the norm of the gradient vector of all trainable params.
                # max_grad_norm=1.0 is standard for transformer fine-tuning;
                # prevents occasional large gradient spikes from corrupting
                # the pre-trained LoRA weights early in training.
                all_params = conditioner_params + lora_params
                torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / cfg.grad_accum_steps
                running_loss = 0.0
                epoch_loss += avg_loss
                n_batches += 1

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr_lora": f"{scheduler.get_last_lr()[-1]:.2e}",
                    "step": global_step,
                })

                # --------------------------------------------------
                # Logging
                # --------------------------------------------------
                if global_step % cfg.log_every_steps == 0 and cfg.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr_conditioner": optimizer.param_groups[0]["lr"],
                        "train/lr_lora": optimizer.param_groups[1]["lr"],
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    }, step=global_step)

                # --------------------------------------------------
                # Validation
                # --------------------------------------------------
                if global_step % cfg.val_every_steps == 0:
                    val_loss = run_validation(model, val_loader, device, use_bf16)
                    print(f"\n  [Step {global_step}] Val loss: {val_loss:.4f} "
                          f"(best: {best_val_loss:.4f})")

                    if cfg.use_wandb:
                        import wandb
                        wandb.log({"val/loss": val_loss}, step=global_step)

                    # Checkpoint
                    ckpt_manager.save(model, optimizer, scheduler, global_step, val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.early_stopping_patience:
                            print(f"\nEarly stopping: val loss did not improve for "
                                  f"{cfg.early_stopping_patience} evaluations.")
                            print(f"Best val loss: {best_val_loss:.4f}")
                            if cfg.use_wandb:
                                import wandb
                                wandb.finish()
                            return

                # --------------------------------------------------
                # Periodic checkpoint (independent of validation)
                # --------------------------------------------------
                elif global_step % cfg.save_every_steps == 0:
                    ckpt_manager.save(model, optimizer, scheduler, global_step)

        # End of epoch summary
        if n_batches > 0:
            print(f"\nEpoch {epoch+1} — mean train loss: {epoch_loss/n_batches:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VarianceEngine — Training — Step 6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to YAML training config.",
    )
    # Any additional args are treated as dot-notation overrides:
    # e.g. --override training.batch_size=2
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in format section.key=value",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg)
