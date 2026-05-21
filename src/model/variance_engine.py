"""
variance_engine.py — VarianceEngine Model (Steps 3–5)

Project        : VarianceEngine
Pipeline stage : Steps 3–5 of 9 (see PIPELINE.md)

Purpose
-------
Wraps MusicGen-medium with:
  1. A frozen EnCodec compression model (encode/decode audio)
  2. AudioEmbeddingConditioner — encodes GT audio to cross-attention context
  3. LoRA-adapted MusicGen transformer — generates variation EnCodec tokens
     conditioned on the GT audio embedding

Architecture overview
---------------------
                   ┌─────────────────────────────┐
  GT audio ──────► │  EnCodec encoder (frozen)   │ ──► (B, D_enc, T_gt)
                   └─────────────────────────────┘
                             │ transpose
                             ▼
                   ┌─────────────────────────────┐
                   │  Projection + LN (trained)  │ ──► (B, T_gt, D_model)
                   └─────────────────────────────┘
                             │ cross-attn ctx
                             ▼
  Var tokens ──► token emb ► ┌─────────────────────────────┐
  (teacher forcing)          │  MusicGen Transformer       │
                             │  + LoRA rank=32 (trained)   │ ──► logits per codebook
                             └─────────────────────────────┘
                             │ EnCodec decoder (frozen)
                             ▼
                         Generated audio

Training objective
------------------
Cross-entropy over all 4 EnCodec codebooks (flat multi-codebook CE loss),
computed with masking over padding positions. Teacher forcing: variation
tokens shifted by 1 are used as input; the model predicts the next token
at each position.

Inference
---------
Nucleus sampling (top-p=0.9) with per-variation random seed. Temperature
is varied uniformly across N requested variations to produce a spectrum of
variation magnitudes (see PIPELINE.md §8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes

from .conditioning import AudioEmbeddingConditioner
from .lora_utils import apply_lora, get_lora_target_modules, print_parameter_summary


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VarianceEngineModel(nn.Module):
    """VarianceEngine: GT-conditioned musical variation generator.

    Parameters
    ----------
    musicgen_name : str
        HuggingFace model ID.  "facebook/musicgen-medium" — 1.5B params.
        Using the base (non-melody) variant because:
        - Its cross-attention layers are used for text; we repurpose them
          for audio GT conditioning, which requires fewer architectural changes
          than adding new cross-attention to a model without any.
        - The melody variant's ChromaStemConditioner extracts only dominant f0,
          which we would need to disable and replace anyway.
    lora_rank : int
        LoRA rank for attention adapters. Default 32 (see PIPELINE.md §5.2).
    conditioner_dropout : float
        Conditioning dropout probability on the AudioEmbeddingConditioner.
        Enables classifier-free guidance at inference if desired.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        musicgen_name: str = "facebook/musicgen-medium",
        lora_rank: int = 32,
        conditioner_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device_str = device

        # ------------------------------------------------------------------
        # 1. Load MusicGen backbone
        # ------------------------------------------------------------------
        print(f"Loading {musicgen_name}...")
        mg = MusicGen.get_pretrained(musicgen_name, device=device)

        # Freeze everything first — LoRA and conditioner unfreeze selectively
        for p in mg.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------
        # 2. Compression model (EnCodec) — fully frozen
        # ------------------------------------------------------------------
        # compression_model: EncodecModel
        #   .encoder  : SEANetEncoder  — encodes audio → continuous embeddings
        #   .quantizer: RVQ            — quantises to discrete codes
        #   .decoder  : SEANetDecoder  — decodes codes → audio
        self.compression_model = mg.compression_model.to(device)
        self.compression_model.eval()
        for p in self.compression_model.parameters():
            p.requires_grad_(False)

        # Key dimensions read from the loaded model
        self.sample_rate: int = mg.sample_rate                    # 32000
        self.n_q: int = mg.lm.n_q                                 # 4 codebooks
        self.card: int = mg.lm.card                               # vocab per codebook (2048)
        # EnCodec encoder output dimension (channels before quantisation)
        self.d_encodec: int = self.compression_model.encoder.dimension  # 128
        # MusicGen transformer hidden dimension
        self.d_model: int = mg.lm.transformer.dim                 # 1024

        # ------------------------------------------------------------------
        # 3. Audio embedding conditioner — replaces text/melody conditioning
        # ------------------------------------------------------------------
        # The conditioner's projection layer is randomly initialised and
        # fully trained. The encoder inside it is the frozen EnCodec encoder.
        self.conditioner = AudioEmbeddingConditioner(
            encodec_encoder=self.compression_model.encoder,
            d_encodec=self.d_encodec,
            d_model=self.d_model,
            dropout=conditioner_dropout,
        ).to(device)

        # ------------------------------------------------------------------
        # 4. LM components needed for forward pass
        # ------------------------------------------------------------------
        # Token embeddings: list of nn.Embedding(card+1, d_model) per codebook
        # +1 for the padding/mask token used at sequence start
        self.emb = mg.lm.emb  # already frozen
        # Output projection: list of nn.Linear(d_model, card) per codebook
        self.linears = mg.lm.linears  # already frozen
        # Transformer backbone — LoRA applied next
        self.transformer = mg.lm.transformer

        # ------------------------------------------------------------------
        # 5. Apply LoRA to transformer attention layers
        # ------------------------------------------------------------------
        print("Discovering LoRA target modules...")
        target_modules = get_lora_target_modules(self.transformer)
        print(f"  Target modules: {target_modules}")

        if not target_modules:
            raise RuntimeError(
                "No attention linear layers found for LoRA. "
                "Check audiocraft version or _ATTN_LAYER_PATTERNS in lora_utils.py."
            )

        self.transformer = apply_lora(
            self.transformer,
            rank=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=target_modules,
        )

        print_parameter_summary(self, "VarianceEngine (full model)")
        print_parameter_summary(self.conditioner, "AudioEmbeddingConditioner")
        print_parameter_summary(self.transformer, "Transformer (after LoRA)")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_audio_to_codes(self, audio: Tensor) -> Tensor:
        """Encode waveform to EnCodec discrete codes (for variation tokens).

        Parameters
        ----------
        audio : Tensor[B, 1, T] at self.sample_rate Hz.

        Returns
        -------
        codes : Tensor[B, n_q, T_frames]
            Discrete token indices per codebook.
        """
        codes, _ = self.compression_model.encode(audio)
        return codes

    @torch.no_grad()
    def decode_codes_to_audio(self, codes: Tensor) -> Tensor:
        """Decode EnCodec codes back to waveform.

        Parameters
        ----------
        codes : Tensor[B, n_q, T_frames]

        Returns
        -------
        Tensor[B, 1, T]
        """
        return self.compression_model.decode(codes, None)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        gt_audio: Tensor,
        var_codes: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute logits for training (teacher forcing).

        Parameters
        ----------
        gt_audio : Tensor[B, 1, T_gt]
            Ground-truth audio at 32kHz, peak-normalised.
        var_codes : Tensor[B, n_q, T_var]
            Variation token codes from EnCodec quantiser.
        attention_mask : Tensor[B, T_var] | None
            1 for valid positions, 0 for padding. Used for masked CE loss.

        Returns
        -------
        logits : Tensor[B, n_q, card, T_var]
            Raw logits per codebook per position. Shift is handled in the
            loss function (input = codes[:, :, :-1], target = codes[:, :, 1:]).

        Implementation notes
        --------------------
        The input to the transformer is the sum of per-codebook token
        embeddings, consistent with MusicGen's original training procedure
        (Copet et al., 2024). This codebook-sum embedding is a deliberate
        design choice in MusicGen: rather than concatenating codebook
        embeddings (which would scale input dim with n_q), summing preserves
        the original model dimension and allows all codebooks to contribute
        equally to the input representation.
        """
        B, n_q, T = var_codes.shape

        # 1. Encode GT audio → conditioning context
        gt_cond = self.conditioner(gt_audio)  # (B, T_gt_frames, D_model)

        # 2. Build input: sum of token embeddings across codebooks
        #    var_codes[:, :, :-1] — teacher forcing: input is all but last token
        input_codes = var_codes[:, :, :-1]  # (B, n_q, T-1)
        # Sum embeddings: (B, T-1, D_model)
        x = sum(self.emb[k](input_codes[:, k]) for k in range(self.n_q))

        # 3. Forward through LoRA-adapted transformer
        #    GT conditioning is passed as cross-attention source.
        #    The transformer's cross-attention layers attend over gt_cond
        #    at every self-attention layer.
        out = self.transformer(x, cross_attention_src=gt_cond)  # (B, T-1, D_model)

        # 4. Project to per-codebook logits
        #    logits[k]: (B, T-1, card) → stack → (B, n_q, T-1, card)
        logits = torch.stack(
            [self.linears[k](out) for k in range(self.n_q)], dim=1
        )  # (B, n_q, T-1, card)

        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        logits: Tensor,
        var_codes: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-entropy loss over all codebooks with optional masking.

        Parameters
        ----------
        logits : Tensor[B, n_q, T-1, card]
        var_codes : Tensor[B, n_q, T]
            Targets are codes[:, :, 1:] (shifted by 1 for next-token prediction).
        attention_mask : Tensor[B, T] | None
            Mask for the original sequence. The loss mask is derived as
            attention_mask[:, 1:] to align with the shifted targets.

        Returns
        -------
        Scalar loss tensor.

        Notes on flat multi-codebook CE
        --------------------------------
        Loss is averaged equally across all 4 codebooks and all valid token
        positions. An alternative is to weight codebooks by their information
        content (codebook 1 contributes most, codebook 4 least). However,
        equal weighting is chosen here because:
        (a) All codebooks contribute to the final audio quality.
        (b) Unequal weighting introduces a hyperparameter without clear
            validation signal at this dataset scale.
        (c) Equal weighting is the standard in MusicGen's original training.
        """
        B, n_q, T_minus1, card = logits.shape
        targets = var_codes[:, :, 1:]  # (B, n_q, T-1)

        # Flatten to (B * n_q * (T-1), card) for F.cross_entropy
        logits_flat = logits.reshape(-1, card)
        targets_flat = targets.reshape(-1)

        if attention_mask is not None:
            # Derive token-level mask: (B, T-1) → (B * n_q * (T-1),)
            mask = attention_mask[:, 1:]  # (B, T-1)
            mask = mask.unsqueeze(1).expand(B, n_q, T_minus1)
            mask_flat = mask.reshape(-1).bool()
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
        else:
            loss = F.cross_entropy(logits_flat, targets_flat)

        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        gt_audio: Tensor,
        n_variations: int = 5,
        max_new_tokens: int = 750,
        top_p: float = 0.9,
        temperature_range: tuple[float, float] = (0.8, 1.2),
        seed: Optional[int] = None,
    ) -> list[Tensor]:
        """Generate N variations of a ground-truth audio clip.

        Parameters
        ----------
        gt_audio : Tensor[1, 1, T]
            Single ground-truth clip. Batch size must be 1.
        n_variations : int
            Number of variations to generate.
        max_new_tokens : int
            Maximum EnCodec frames to generate. At 50 Hz, 750 frames = 15s.
            Default matches the 15s upper bound from preprocessing (Step 2).
        top_p : float
            Nucleus sampling probability mass threshold (see PIPELINE.md §8.3).
            0.9: sample from tokens covering 90% of probability mass.
        temperature_range : tuple[float, float]
            Temperature is sampled uniformly from this range for each variation,
            producing a spectrum from conservative (low T) to expressive (high T).
            Lower bound 0.8 stays in the high-probability region of the
            learned distribution; upper bound 1.2 allows moderate divergence
            without producing incoherent token sequences.
        seed : int | None
            Base random seed. Variation i uses seed + i for reproducibility.

        Returns
        -------
        list of Tensor[1, 1, T_audio] — one decoded waveform per variation.

        Notes on nucleus sampling
        -------------------------
        Nucleus (top-p) sampling is chosen over greedy decoding (produces
        identical N outputs) and unrestricted temperature sampling (risks
        incoherent tokens at high temperature). By restricting sampling to
        the 90% probability nucleus, we prevent degenerate low-probability
        tokens while preserving sufficient diversity for musically plausible
        variation — consistent with MusicGen's own inference defaults
        (Copet et al., 2024).
        """
        self.eval()
        assert gt_audio.shape[0] == 1, "generate() expects batch size 1"

        gt_cond = self.conditioner(gt_audio)  # (1, T_gt_frames, D_model)

        # Temperature values evenly spaced across the range for N variations
        import numpy as np
        temperatures = np.linspace(
            temperature_range[0], temperature_range[1], n_variations
        ).tolist()

        generated_waveforms = []

        for i, temp in enumerate(temperatures):
            if seed is not None:
                torch.manual_seed(seed + i)

            # Autoregressive generation: start from BOS token (index = card)
            # Shape: (1, n_q, 1) — one BOS token per codebook
            bos = torch.full(
                (1, self.n_q, 1),
                fill_value=self.card,
                dtype=torch.long,
                device=gt_audio.device,
            )
            tokens = bos  # (1, n_q, current_len)

            for _ in range(max_new_tokens):
                # Embed current sequence
                x = sum(
                    self.emb[k](tokens[:, k]) for k in range(self.n_q)
                )  # (1, current_len, D_model)

                # Forward pass
                out = self.transformer(x, cross_attention_src=gt_cond)
                # Logits for next token only (last position)
                next_logits = torch.stack(
                    [self.linears[k](out[:, -1, :]) for k in range(self.n_q)],
                    dim=1,
                )  # (1, n_q, card)

                # Sample next token per codebook via nucleus sampling
                next_tokens = _nucleus_sample(next_logits, top_p=top_p, temperature=temp)
                # (1, n_q, 1)

                tokens = torch.cat([tokens, next_tokens], dim=2)

            # Strip BOS token, decode to audio
            codes = tokens[:, :, 1:]  # (1, n_q, max_new_tokens)
            waveform = self.decode_codes_to_audio(codes)  # (1, 1, T)
            generated_waveforms.append(waveform.cpu())

        return generated_waveforms

    # ------------------------------------------------------------------
    # Checkpoint save / load (LoRA + conditioner only)
    # ------------------------------------------------------------------

    def save_trainable_weights(self, path: str | Path) -> None:
        """Save only trainable parameters (LoRA adapters + conditioner).

        The base MusicGen weights are not saved — they are loaded from
        HuggingFace at inference time. This keeps checkpoints small (~50 MB
        for LoRA rank=32) and avoids redistributing the CC-BY-NC model weights.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trainable_state = {
            k: v for k, v in self.state_dict().items()
            if any(
                k.startswith(prefix)
                for prefix in ("conditioner.projection", "conditioner.norm",
                               "transformer.base_model.model")  # LoRA weights via PEFT
            ) and "lora_" in k or k.startswith("conditioner.")
        }

        # Include full conditioner state
        conditioner_state = {
            f"conditioner.{k}": v
            for k, v in self.conditioner.state_dict().items()
        }

        torch.save(
            {
                "trainable_weights": trainable_state,
                "conditioner": conditioner_state,
                "config": {
                    "d_encodec": self.d_encodec,
                    "d_model": self.d_model,
                    "n_q": self.n_q,
                    "card": self.card,
                    "sample_rate": self.sample_rate,
                },
            },
            path,
        )
        print(f"Saved trainable weights → {path}")

    @classmethod
    def from_pretrained_weights(
        cls,
        checkpoint_path: str | Path,
        musicgen_name: str = "facebook/musicgen-medium",
        lora_rank: int = 32,
        device: str = "cuda",
    ) -> "VarianceEngineModel":
        """Load a VarianceEngine from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(musicgen_name=musicgen_name, lora_rank=lora_rank, device=device)

        # Load conditioner weights
        conditioner_state = {
            k.replace("conditioner.", ""): v
            for k, v in checkpoint["conditioner"].items()
        }
        model.conditioner.load_state_dict(conditioner_state, strict=False)

        # Load LoRA weights
        missing, unexpected = model.load_state_dict(
            checkpoint["trainable_weights"], strict=False
        )
        if unexpected:
            print(f"  Warning: unexpected keys in checkpoint: {unexpected[:5]}")

        print(f"Loaded VarianceEngine weights from {checkpoint_path}")
        return model


# ---------------------------------------------------------------------------
# Sampling utility
# ---------------------------------------------------------------------------

def _nucleus_sample(logits: Tensor, top_p: float, temperature: float) -> Tensor:
    """Apply nucleus (top-p) sampling to per-codebook logits.

    Parameters
    ----------
    logits : Tensor[B, n_q, card]
    top_p : float
        Retain tokens covering top_p probability mass.
    temperature : float
        Scales logits before softmax. >1 increases diversity; <1 sharpens.

    Returns
    -------
    Tensor[B, n_q, 1] — sampled token indices.

    Algorithm
    ---------
    1. Scale logits by 1/temperature.
    2. Convert to probabilities.
    3. Sort descending; compute cumulative sum.
    4. Zero out tokens beyond the top_p nucleus.
    5. Renormalise and sample.

    This implementation follows the original nucleus sampling formulation
    (Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration").
    """
    B, n_q, card = logits.shape
    logits = logits / max(temperature, 1e-5)

    probs = torch.softmax(logits, dim=-1)  # (B, n_q, card)

    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p
    # (shift by 1 to include the token that pushes cumsum over top_p)
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_indices_to_remove] = 0.0

    # Renormalise
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Sample
    # Reshape to (B * n_q, card) for multinomial
    flat_probs = sorted_probs.reshape(B * n_q, card)
    flat_sampled = torch.multinomial(flat_probs, num_samples=1)  # (B*n_q, 1)

    # Map back from sorted indices to original indices
    flat_sorted_indices = sorted_indices.reshape(B * n_q, card)
    flat_next_token = flat_sorted_indices.gather(1, flat_sampled)  # (B*n_q, 1)

    next_token = flat_next_token.reshape(B, n_q, 1)
    return next_token
