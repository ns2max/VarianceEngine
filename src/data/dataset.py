"""
dataset.py — Step 6: Training Dataset

Project        : VarianceEngine
Pipeline stage : Step 6 of 9 (see PIPELINE.md)

Purpose
-------
PyTorch Dataset and DataLoader utilities for the VarianceEngine training loop.

Each dataset item is a (ground_truth_audio, variation_audio) pair from
outputs/split_pairs.json.  Audio is returned as float32 tensors at 32kHz;
EnCodec encoding to discrete codes happens in the training loop on GPU to
avoid CUDA context issues in DataLoader workers.

Design decisions
----------------

Random crop for long files (> max_duration_s):
  Applied independently to GT and variation audio during training.
  At evaluation time, a deterministic centre crop is used for reproducibility.
  Rationale: random cropping augments the dataset with different phrase
  segments from long patterns, preventing the model from always seeing the
  same temporal context. Centre crop at eval ensures consistent comparisons.

  GT and variation crops are NOT synchronised (i.e. the same time offset is
  not used for both). GT and variation are different performances recorded
  independently; there is no meaningful temporal alignment between them.

Variable-length batching via custom collate:
  Variation audio tensors are right-zero-padded to the length of the longest
  item in the batch. An attention mask tracks valid frames at the EnCodec code
  level (50 Hz). Padding positions are excluded from the CE loss.

  GT audio is also padded to the longest GT in the batch. Because the GT
  conditioner operates on the full sequence (the transformer attends over ALL
  GT frames via cross-attention), GT padding does not require a separate mask
  — the padded frames produce near-zero encoder activations which the
  cross-attention weights effectively learn to ignore.

DataLoader workers:
  num_workers=0 because EnCodec encoding in __getitem__ would require CUDA,
  which cannot be used in forked worker processes. All audio loading is done
  in the main process. For faster I/O on the server, use num_workers=2 with
  pin_memory=True if audio loading is a bottleneck (requires profiling).
"""

import json
import random
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EnCodec hop size in samples at 32kHz → 50 Hz frame rate
_ENCODEC_HOP = 640  # 32000 / 50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VariancePairDataset(Dataset):
    """Dataset of (ground_truth_audio, variation_audio) pairs.

    Parameters
    ----------
    pairs : list[dict]
        From outputs/split_pairs.json. Each dict has keys:
        'gt_preprocessed', 'var_preprocessed', 'artist_id', 'pattern_id'.
    sample_rate : int
        Expected sample rate of preprocessed files (32000).
    max_duration_s : float
        Hard cap per file. Files longer than this are cropped.
    split : str
        'train' → random crop; 'val'/'test' → centre crop.
    """

    def __init__(
        self,
        pairs: list[dict],
        sample_rate: int = 32_000,
        max_duration_s: float = 15.0,
        split: str = "train",
    ):
        self.pairs = pairs
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)
        self.is_train = split == "train"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        gt_audio = self._load_and_crop(pair["gt_preprocessed"])
        var_audio = self._load_and_crop(pair["var_preprocessed"])

        return {
            "gt_audio": gt_audio,               # (1, T_gt)
            "var_audio": var_audio,             # (1, T_var)
            "artist_id": pair["artist_id"],
            "pattern_id": pair["pattern_id"],
        }

    def _load_and_crop(self, filepath: str) -> Tensor:
        """Load a preprocessed WAV and apply duration cap.

        Returns
        -------
        Tensor[1, T] — mono float32 waveform.
        """
        # librosa confirmed working on this dataset (Steps 1, 2).
        # sr=None uses the file's native rate (32kHz after preprocessing).
        audio, _ = librosa.load(filepath, sr=None, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

        if audio.shape[-1] > self.max_samples:
            if self.is_train:
                # Random crop: augments with different temporal phrase segments
                start = random.randint(0, audio.shape[-1] - self.max_samples)
            else:
                # Centre crop: deterministic for reproducible val/test metrics
                start = (audio.shape[-1] - self.max_samples) // 2
            audio = audio[:, start : start + self.max_samples]

        return audio


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch of (gt_audio, var_audio) pairs into padded tensors.

    GT audio: right-zero-padded to the longest GT in the batch.
    Var audio: right-zero-padded to the longest variation in the batch.

    Attention mask (var):
        Boolean tensor[B, T_var_frames] at EnCodec frame rate (50 Hz).
        True = valid frame; False = padding frame.
        The mask is used to exclude padded frames from CE loss computation.

    GT does not have an explicit mask — cross-attention over padded GT frames
    produces near-zero activations (the padding is zero-valued) which the
    attention weights learn to down-weight. Adding a GT mask would require
    modifying the transformer's cross-attention, adding complexity with
    marginal benefit given that typical GT padding is <10% of sequence length.

    Returns
    -------
    dict with keys:
        gt_audio       : Tensor[B, 1, T_gt_max]
        var_audio      : Tensor[B, 1, T_var_max]
        attention_mask : Tensor[B, T_var_frames]   (True = valid)
        artist_ids     : list[int]
        pattern_ids    : list[int]
    """
    gt_audios  = [item["gt_audio"]  for item in batch]
    var_audios = [item["var_audio"] for item in batch]

    # Record original var lengths for mask construction
    var_lengths = [v.shape[-1] for v in var_audios]

    # Pad GT
    max_gt = max(g.shape[-1] for g in gt_audios)
    gt_padded = torch.stack([
        F.pad(g, (0, max_gt - g.shape[-1])) for g in gt_audios
    ])  # (B, 1, T_gt_max)

    # Pad var
    max_var = max(var_lengths)
    var_padded = torch.stack([
        F.pad(v, (0, max_var - v.shape[-1])) for v in var_audios
    ])  # (B, 1, T_var_max)

    # Attention mask at EnCodec frame level
    # n_frames for a waveform of length L = ceil(L / _ENCODEC_HOP)
    max_frames = (max_var + _ENCODEC_HOP - 1) // _ENCODEC_HOP
    attention_mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    for i, length in enumerate(var_lengths):
        n_valid_frames = (length + _ENCODEC_HOP - 1) // _ENCODEC_HOP
        attention_mask[i, :n_valid_frames] = True

    return {
        "gt_audio": gt_padded,
        "var_audio": var_padded,
        "attention_mask": attention_mask,
        "artist_ids": [item["artist_id"] for item in batch],
        "pattern_ids": [item["pattern_id"] for item in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    split_pairs_path: str,
    sample_rate: int = 32_000,
    max_duration_s: float = 15.0,
    batch_size: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from split_pairs.json.

    Parameters
    ----------
    split_pairs_path : str
        Path to outputs/split_pairs.json produced by preprocess.py.
    sample_rate : int
        Expected sample rate of preprocessed files.
    max_duration_s : float
        Hard duration cap (applied per file).
    batch_size : int
        Per-GPU batch size.
    seed : int
        Random seed for DataLoader worker init (reproducibility).

    Returns
    -------
    (train_loader, val_loader, test_loader)

    DataLoader configuration notes
    -------------------------------
    shuffle=True for train: standard practice; ensures different batch
    compositions each epoch, preventing the model from memorising pair order.

    num_workers=0: EnCodec encoding happens on GPU in the training loop,
    not in the Dataset. Dataset only does audio file I/O + cropping, which
    is fast enough in the main process for this dataset size (~1,400 pairs).
    If audio loading becomes a bottleneck (profile first), set num_workers=2
    with pin_memory=True and ensure audio loading stays CPU-only in workers.

    drop_last=True for train: prevents a final incomplete batch from
    distorting gradient estimates when batch size is small (4). The dropped
    samples rotate each epoch due to shuffle.
    """
    with open(split_pairs_path) as f:
        split_pairs = json.load(f)

    def _make_loader(pairs: list[dict], split: str) -> DataLoader:
        dataset = VariancePairDataset(
            pairs=pairs,
            sample_rate=sample_rate,
            max_duration_s=max_duration_s,
            split=split,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=(split == "train"),
            pin_memory=False,  # num_workers=0 → pin_memory has no effect
            worker_init_fn=lambda wid: random.seed(seed + wid),
        )

    train_loader = _make_loader(split_pairs["train"], "train")
    val_loader   = _make_loader(split_pairs["val"],   "val")
    test_loader  = _make_loader(split_pairs["test"],  "test")

    print(f"DataLoaders built:")
    print(f"  Train: {len(split_pairs['train'])} pairs, {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(split_pairs['val'])} pairs,  {len(val_loader)} batches")
    print(f"  Test:  {len(split_pairs['test'])} pairs, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
