"""
lora_utils.py — LoRA configuration and parameter accounting utilities.

Project        : VarianceEngine
Pipeline stage : Step 5 of 9 (see PIPELINE.md)

Purpose
-------
Configures LoRA (Low-Rank Adaptation) for MusicGen's transformer attention
layers via the PEFT library, and provides utilities for auditing trainable
parameter counts.

Architecture Decision (from PIPELINE.md §5.2)
----------------------------------------------
LoRA at rank=32 is applied to existing attention Q/K/V/O projections.
New conditioning layers (AudioEmbeddingConditioner projection + norm) are
fully trained — they have no pre-trained weights to protect.

Rank=32 chosen over rank=16:
  The variation distribution spans both ornamental and structural pitch changes.
  Ornamental changes require fine local weight adjustments; structural changes
  (full transpositions, mode changes) require the update matrix to cover a
  wider subspace. Rank=32 provides ~8M additional parameters in attention
  layers, sufficient to span both transformation types while remaining within
  the 4090's VRAM budget (~9 GB with bf16 + gradient checkpointing).

Target modules:
  MusicGen's StreamingTransformer uses nn.Linear layers for Q/K/V/O projections
  within StreamingMultiheadAttention. PEFT targets these by module name suffix.
  The exact names are discovered at runtime via `get_lora_target_modules()` to
  guard against audiocraft version differences.

Frozen components:
  EnCodec encoder/decoder, compression model quantiser, original text
  conditioning pathway (disabled), non-attention transformer weights (FFN,
  layer norms). These retain their pre-trained music knowledge unchanged.
"""

import re
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Target module discovery
# ---------------------------------------------------------------------------

# Patterns that match attention projection layers in audiocraft's transformer.
# These cover the most common audiocraft naming conventions across versions.
_ATTN_LAYER_PATTERNS = [
    r".*self_attn\.q_proj$",
    r".*self_attn\.k_proj$",
    r".*self_attn\.v_proj$",
    r".*self_attn\.out_proj$",
    r".*self_attn\.in_proj$",        # fused QKV projection (some versions)
    r".*cross_attention\.q_proj$",
    r".*cross_attention\.k_proj$",
    r".*cross_attention\.v_proj$",
    r".*cross_attention\.out_proj$",
]


def get_lora_target_modules(model: nn.Module) -> list[str]:
    """Discover attention linear layer names in a MusicGen transformer.

    Walks the model's named modules and returns the name suffixes that match
    known attention projection patterns. These are passed to PEFT's LoraConfig
    as `target_modules`.

    Returns a deduplicated list of module name suffixes (e.g. ['q_proj',
    'k_proj', 'v_proj', 'out_proj']) rather than full dotted paths, because
    PEFT matches by suffix across all layers.

    Parameters
    ----------
    model : nn.Module
        The MusicGen LM model (lm attribute of MusicGen).

    Returns
    -------
    list[str] — unique module name suffixes to target with LoRA.
    """
    found_suffixes: set[str] = set()
    compiled = [re.compile(p) for p in _ATTN_LAYER_PATTERNS]

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for pattern in compiled:
            if pattern.match(name):
                suffix = name.split(".")[-1]
                found_suffixes.add(suffix)
                break

    if not found_suffixes:
        # Fallback: target any Linear layer in self_attn or cross_attention
        # This is a defensive fallback for unexpected audiocraft versions.
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if "self_attn" in name or "cross_attention" in name:
                    found_suffixes.add(name.split(".")[-1])

    return sorted(found_suffixes)


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

def apply_lora(
    lm_model: nn.Module,
    rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """Apply LoRA to MusicGen's language model transformer attention layers.

    Parameters
    ----------
    lm_model : nn.Module
        MusicGen's lm (language model) component.
    rank : int
        LoRA rank. Default 32 (see PIPELINE.md §5.2 for rationale).
    lora_alpha : int
        LoRA scaling factor. Convention: set to 2× rank so the effective
        learning rate scaling (alpha/rank = 2) is consistent regardless of rank.
    lora_dropout : float
        Dropout on LoRA A/B matrices. Light regularisation for small datasets.
    target_modules : list[str] | None
        Module name suffixes to apply LoRA to. If None, auto-discovered via
        `get_lora_target_modules()`.

    Returns
    -------
    nn.Module — the lm_model with LoRA adapters inserted (PEFT PeftModel).

    Notes on lora_alpha
    -------------------
    The LoRA update is scaled by alpha/rank. Setting alpha=2*rank keeps this
    ratio constant at 2.0 independent of rank choice, so the effective step
    size is comparable across the rank ablations (rank 16/32/64) in Step 7.3.
    This follows the convention established in Hu et al. (2022).

    Notes on lora_dropout
    ---------------------
    0.05 (5%) dropout on the LoRA matrices is light regularisation appropriate
    for a dataset of ~1,800 training pairs. It prevents the low-rank adapter
    from overfitting to the most frequent variation patterns while remaining
    small enough not to destabilise gradient flow through the adapters.
    """
    if target_modules is None:
        target_modules = get_lora_target_modules(lm_model)

    if not target_modules:
        raise RuntimeError(
            "No LoRA target modules found in model. "
            "Check that the model has attention linear layers matching "
            "patterns in lora_utils._ATTN_LAYER_PATTERNS."
        )

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        # PEFT does not have a built-in TaskType for audio generation;
        # CAUSAL_LM is the closest structural match (autoregressive token
        # prediction with cross-entropy loss) and does not affect the
        # LoRA weight structure.
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(lm_model, config)
    return peft_model


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """Count and categorise trainable vs frozen parameters.

    Returns
    -------
    dict with keys:
        trainable       : int — parameters with requires_grad=True
        frozen          : int — parameters with requires_grad=False
        total           : int
        trainable_pct   : float — percentage of total
        trainable_M     : float — trainable params in millions
        total_M         : float — total params in millions
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen

    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_pct": round(100 * trainable / total, 3) if total > 0 else 0.0,
        "trainable_M": round(trainable / 1e6, 2),
        "total_M": round(total / 1e6, 2),
    }


def print_parameter_summary(model: nn.Module, label: str = "Model") -> None:
    """Print a human-readable parameter summary to stdout."""
    stats = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  {label} — Parameter Summary")
    print(f"{'='*50}")
    print(f"  Total      : {stats['total_M']:.1f}M")
    print(f"  Trainable  : {stats['trainable_M']:.1f}M  ({stats['trainable_pct']:.2f}%)")
    print(f"  Frozen     : {round(stats['frozen']/1e6, 1):.1f}M")
    print(f"{'='*50}\n")
