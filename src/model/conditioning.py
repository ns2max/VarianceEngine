"""
conditioning.py — Step 4: Audio Embedding Conditioner

Project        : VarianceEngine
Pipeline stage : Step 4 of 9 (see PIPELINE.md)

Purpose
-------
Implements the AudioEmbeddingConditioner: the module that encodes a ground-truth
WAV into a temporally-aligned conditioning tensor consumed by the MusicGen
transformer's cross-attention layers.

Architecture Decision (from PIPELINE.md §4)
--------------------------------------------
Input audio is encoded via the frozen EnCodec encoder (pre-quantization)
to produce continuous embeddings at 50 frames/second. These are projected to
the transformer model dimension via a learned linear layer. The continuous
(pre-quantization) embeddings are chosen over discrete tokens because they
retain soft probability mass across all codebook dimensions simultaneously —
critical for polyphonic audio where multiple pitch classes are active per frame.

Rejected alternative: CLAP audio embedding
  CLAP compresses the full audio to a single 512-d vector, discarding temporal
  structure. For patterns where harmonic content evolves over ~5s, this is an
  unacceptable information loss. EnCodec at 50Hz provides ~250 conditioning
  frames for a 5s input — preserving phrase-level harmonic trajectories.

Rejected alternative: Post-quantization discrete tokens
  Discrete tokens from codebook 1 cannot represent simultaneous chord tones;
  the quantiser forces a nearest-neighbour assignment that collapses polyphony.
  Continuous embeddings carry the full pre-quantisation activation.
"""

import torch
import torch.nn as nn
from torch import Tensor


class AudioEmbeddingConditioner(nn.Module):
    """Encodes ground-truth audio into a cross-attention conditioning tensor.

    Parameters
    ----------
    encodec_encoder : nn.Module
        The frozen SEANetEncoder from MusicGen's compression model.
        Call signature: encoder(x: Tensor[B, 1, T]) -> Tensor[B, D_enc, T_frames]
    d_encodec : int
        Output channel dimension of the EnCodec encoder (128 for MusicGen).
    d_model : int
        Transformer hidden dimension (1024 for MusicGen-medium).
    dropout : float
        Applied to projected conditioning embeddings during training.
        Implements a form of conditioning dropout — with probability `dropout`
        the conditioner output is zeroed, which implicitly trains the model for
        both conditional and unconditional generation. This enables
        classifier-free guidance at inference time if desired.

    Forward input
    -------------
    gt_audio : Tensor[B, 1, T]
        Ground-truth audio at 32kHz, peak-normalised.

    Forward output
    --------------
    Tensor[B, T_frames, D_model] — conditioning context for cross-attention.
    T_frames = ceil(T / hop_length), approximately 50 * duration_seconds.
    """

    def __init__(
        self,
        encodec_encoder: nn.Module,
        d_encodec: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # EnCodec encoder is always frozen — it is not fine-tuned.
        # Its pre-trained representation is used as-is; only the projection
        # layer learns to select which aspects of the encoding to condition on.
        self.encoder = encodec_encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # MLP projection: EnCodec dim → hidden → transformer dim.
        #
        # Upgraded from single Linear to two-layer MLP with GELU activation.
        # Rationale: the frozen cross-attention W_k and W_v were pre-trained on
        # T5 text embeddings (1024-dim, specific statistical distribution). Our
        # audio embeddings from EnCodec (128-dim) are in a completely different
        # distribution. A single linear projection (128→1536) cannot reshape the
        # distribution enough for the frozen W_k/W_v to extract useful features.
        # A two-layer MLP with a hidden bottleneck can learn a non-linear mapping
        # that better aligns audio embeddings with the text-embedding distribution
        # the cross-attention attention mechanism was trained on.
        #
        # Hidden dim = d_model (1536): keeps the intermediate representation in
        # the same space as the transformer, avoiding an information bottleneck.
        d_hidden = d_model
        self.proj_in  = nn.Linear(d_encodec, d_hidden, bias=True)
        self.act      = nn.GELU()
        self.proj_out = nn.Linear(d_hidden, d_model, bias=True)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, gt_audio: Tensor) -> Tensor:
        """Encode GT audio to cross-attention conditioning context.

        Parameters
        ----------
        gt_audio : Tensor[B, 1, T]

        Returns
        -------
        Tensor[B, T_frames, D_model]
        """
        with torch.no_grad():
            # Force fp32: EnCodec's SEANetEncoder contains an LSTM layer whose
            # CUDA kernel (_thnn_fused_lstm_cell_cuda) does not support bf16.
            # The outer autocast context in the training loop would otherwise
            # cast inputs to bf16 and cause a runtime error.
            with torch.cuda.amp.autocast(enabled=False):
                enc = self.encoder(gt_audio.float())  # (B, D_enc, T_frames)

        # (B, T_frames, D_enc)
        enc = enc.transpose(1, 2)

        # MLP projection: EnCodec dim → d_model
        cond = self.proj_in(enc)    # (B, T_frames, D_hidden)
        cond = self.act(cond)
        cond = self.proj_out(cond)  # (B, T_frames, D_model)
        cond = self.norm(cond)
        cond = self.dropout(cond)

        return cond
