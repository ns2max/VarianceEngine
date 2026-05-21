"""
lora_utils.py — Manual LoRA implementation (no PEFT dependency).

Project        : VarianceEngine
Pipeline stage : Step 5 of 9 (see PIPELINE.md)

Why manual LoRA instead of PEFT
---------------------------------
PEFT requires transformers >= 4.36 (for the Cache class). audiocraft requires
a pinned transformers version incompatible with recent PEFT. Rather than fight
three-way version pinning (torch / transformers / peft), LoRA is implemented
directly in PyTorch. The implementation is ~50 lines and exactly matches
the mathematical definition in Hu et al. (2022):

    h = W₀x + (B A x) × (α / r)

where W₀ is the frozen pre-trained weight, A ∈ ℝ^{r×d_in} and B ∈ ℝ^{d_out×r}
are the low-rank adapter matrices, α is the scaling factor, and r is the rank.

Architecture Decision (from PIPELINE.md §5.2)
----------------------------------------------
LoRA at rank=32 applied to Q, K, V, O projection Linear layers in the
transformer's self-attention (and cross-attention if present).

lora_alpha = 2 × rank: keeps effective scaling (alpha/rank = 2) constant
across rank ablation experiments (rank 16/32/64 in Step 7.3).

lora_dropout = 0.05: light regularisation on the adapter path appropriate for
~1,400 training pairs. Prevents the adapter from memorising frequent patterns.
"""

import math
import re
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    The original weight W₀ is frozen. Only lora_A and lora_B are trained.

    Initialisation:
        lora_A ~ Kaiming uniform (standard for linear layers)
        lora_B = 0  (ensures the adapter output is zero at init, so the model
                      starts from its pre-trained state — essential for
                      stable fine-tuning)

    Parameters
    ----------
    linear : nn.Linear
        The original pre-trained linear layer to wrap.
    rank : int
        LoRA rank r. Controls adapter expressivity.
    alpha : float
        Scaling factor. Effective scale = alpha / rank.
    dropout : float
        Dropout applied to the input before the adapter path.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank

        in_features  = linear.in_features
        out_features = linear.out_features
        # Must be on the same device as the wrapped layer.
        # lora_A/B are always float32 for numerical stability even in bf16 training.
        device = linear.weight.device

        self.lora_A = nn.Linear(in_features, rank, bias=False).to(device)
        self.lora_B = nn.Linear(rank, out_features, bias=False).to(device)
        self.dropout = nn.Dropout(p=dropout)

        # Kaiming uniform init for A (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # Zero init for B: ensures output delta = 0 at training start
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weight
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        base_out  = self.linear(x)
        lora_out  = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


# ---------------------------------------------------------------------------
# Target module discovery
# ---------------------------------------------------------------------------

# Name suffixes that identify attention projection layers across audiocraft versions.
_ATTN_SUFFIXES = {"q_proj", "k_proj", "v_proj", "out_proj", "in_proj"}


def get_lora_target_modules(model: nn.Module) -> list[str]:
    """Return unique name suffixes of attention Linear layers in the model.

    Walks all named modules, collects the suffix (last dotted component) of
    any nn.Linear whose name contains 'self_attn' or 'cross_attention' and
    whose suffix is in the known attention projection set.

    Falls back to any Linear inside self_attn/cross_attention if none found.
    """
    found: set[str] = set()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        in_attn = "self_attn" in name or "cross_attention" in name
        suffix  = name.split(".")[-1]
        if in_attn and suffix in _ATTN_SUFFIXES:
            found.add(suffix)

    if not found:
        # Fallback: any Linear inside attention blocks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if "self_attn" in name or "cross_attention" in name:
                    found.add(name.split(".")[-1])

    return sorted(found)


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

def apply_lora(
    model: nn.Module,
    rank: int = 32,
    lora_alpha: Optional[float] = None,
    lora_dropout: float = 0.05,
    target_suffixes: Optional[list[str]] = None,
) -> nn.Module:
    """Replace target Linear layers in model with LoRALinear in-place.

    Parameters
    ----------
    model : nn.Module
        The transformer model to modify (MusicGen lm.transformer).
    rank : int
        LoRA rank. Default 32 (see PIPELINE.md §5.2).
    lora_alpha : float | None
        Scaling factor. Defaults to 2 × rank (effective scale = 2.0).
    lora_dropout : float
        Dropout on the adapter input path.
    target_suffixes : list[str] | None
        Layer name suffixes to target. Auto-discovered if None.

    Returns
    -------
    The same model with LoRALinear layers substituted in-place.

    How in-place substitution works
    --------------------------------
    For each matching nn.Linear, we locate its parent module and attribute
    name using the dotted path, then replace it with LoRALinear(original).
    The original weight tensor is preserved inside LoRALinear.linear and
    is frozen (requires_grad=False). The LoRA adapter matrices are trainable.
    """
    if lora_alpha is None:
        lora_alpha = float(rank * 2)

    if target_suffixes is None:
        target_suffixes = get_lora_target_modules(model)

    if not target_suffixes:
        raise RuntimeError(
            "No attention Linear layers found for LoRA. "
            "Check _ATTN_SUFFIXES in lora_utils.py matches the audiocraft version."
        )

    target_set = set(target_suffixes)
    replaced = 0

    # Collect (parent, attr_name, full_name) for all matching layers
    # We cannot modify the model while iterating named_modules(), so collect first.
    to_replace: list[tuple[nn.Module, str, str]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = full_name.split(".")[-1]
        in_attn = "self_attn" in full_name or "cross_attention" in full_name
        if suffix in target_set and in_attn:
            # Find parent
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            to_replace.append((parent, parts[-1], full_name))

    for parent, attr, full_name in to_replace:
        original = getattr(parent, attr)
        lora_layer = LoRALinear(
            linear=original,
            rank=rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        setattr(parent, attr, lora_layer)
        replaced += 1

    print(f"  LoRA: replaced {replaced} Linear layers "
          f"(rank={rank}, alpha={lora_alpha}, dropout={lora_dropout})")
    print(f"  Target suffixes: {sorted(target_set)}")

    return model


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """Count trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen
    return {
        "trainable":    trainable,
        "frozen":       frozen,
        "total":        total,
        "trainable_pct": round(100 * trainable / total, 3) if total else 0.0,
        "trainable_M":  round(trainable / 1e6, 2),
        "total_M":      round(total / 1e6, 2),
    }


def print_parameter_summary(model: nn.Module, label: str = "Model") -> None:
    s = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  {label} — Parameter Summary")
    print(f"{'='*50}")
    print(f"  Total     : {s['total_M']:.1f}M")
    print(f"  Trainable : {s['trainable_M']:.1f}M  ({s['trainable_pct']:.2f}%)")
    print(f"  Frozen    : {round(s['frozen']/1e6, 1):.1f}M")
    print(f"{'='*50}\n")
