# VarianceEngine — Project Memory

> Complete project state log. Sufficient to resume from any point without prior conversation context.
> Last updated: 2026-05-21

---

## Project Overview

**Goal:** Given one ground-truth musical pattern (WAV) and integer N, generate N expressive variations.

**Dataset:** DOPP — ~2,276 mono WAV files, 22kHz, 176 patterns, 20 artists.  
Filename format: `artistID_patternID_versionID.wav` — versionID=0 is ground truth.  
Variations include ornamental (trills, passing tones) and structural (transpositions, chord substitutions, mode changes) pitch/harmonic changes, plus timing and dynamics.

**Approach chosen:** E — fine-tune pre-trained audio foundation model (MusicGen-medium, 1.5B params).

**Hardware:** RTX 4090 (24 GB, cuda:0, primary) + RTX 3090 (24 GB, cuda:1, secondary).  
All compute runs manually on server. This repo contains only code and configs.

**Paper:** Research paper being written. Every decision requires pros/cons + supporting claim documented in PIPELINE.md.

**Project path:** `/Users/nishal/PROJECTS/datasets/VarianceEngine/`  
**Dataset path:** `dopp/` (relative to project root)

---

## Approach Selection

Five approaches were evaluated (documented in `APPROACHES.md`):

| Approach | Method | Rejected because |
|---|---|---|
| A | Rule-based pitch shifting | No learned variation; only transposition |
| B | VAE latent interpolation | No pre-trained audio VAE for this domain |
| C | Retrieval + pitch shift | Cannot generate new variations, only recombine |
| D | Train generative model from scratch | Insufficient data (~2K pairs) for scratch training |
| **E** | Fine-tune MusicGen-medium | **Chosen** — pre-trained harmonic knowledge + audio conditioning |

---

## Dataset Findings (Step 1)

Run: `python src/analyze_dataset.py --data_dir dopp/ --output_dir outputs/ --workers 4`

| Metric | Expected | Actual |
|---|---|---|
| Duration range | 4.6–11.7 s | 0.6–25.9 s |
| Mean duration | ~7.0 s | 5.2 s |
| GT–variation pairs | ~1,600 | 2,100 |
| Clipped files (peak≥0.999) | 0 | 70 across 43 patterns |
| Clipped GT files (versionID=0) | 0 | 4: patterns 51_0, 52_1, 52_8, 54_1 |
| Chroma similarity mean | — | 0.9555 (median 0.9813) |
| Variations with similarity > 0.9 | bimodal expected | 89.3% — heavily ornamental |
| Variations with similarity < 0.7 | substantial | 2.1% only |

**Key finding:** Dataset is NOT bimodal. Heavily ornamental. Structural variation minority (2.1%). Ablation sets for ornamental vs. structural will be severely imbalanced — must report with caveat.

---

## Preprocessing Decisions (Step 2)

Run: `python src/preprocess.py --data_dir dopp/ --stats_path outputs/dataset_stats.json --output_dir preprocessed/ --splits_dir outputs/ --workers 4`

**Clipping handling:**
- Patterns with clipped GT (versionID=0) → entire pattern excluded (4 patterns, ~48 pairs lost)
- Clipped variations within valid patterns → excluded individually
- Threshold: peak_amplitude >= 0.999 (float rounding on 16-bit PCM)

**Duration filtering:**
- Files < 1.0s → excluded (76 files; too few EnCodec tokens for coherent generation)
- Files > 15.0s → NOT excluded; random crop applied at training time

**Resampling:** 22,050 Hz → 32,000 Hz (MusicGen EnCodec native rate) using `torchaudio.functional.resample` with sinc interpolation.

**Normalization:** Peak normalize to -1 dBFS after clipping exclusion (not before — normalizing a clipped file amplifies flat-top distortion).

**Split strategy:** 80/10/10 by `pattern_id`, stratified by `artist_id`. Split by pattern (not file) to prevent data leakage — test set contains patterns never seen in any version during training. Seed=42. Stored in `outputs/splits.json`.

**Outputs:**
- `preprocessed/` — cleaned WAVs at 32kHz
- `outputs/splits.json` — train/val/test pattern lists
- `outputs/split_pairs.json` — GT–variation pairs per split
- `outputs/preprocessing_report.json` — audit of exclusions

---

## Model Architecture (Steps 3–5)

### Foundation Model: MusicGen-medium

- 1.5B parameters, autoregressive transformer + EnCodec codec
- EnCodec: 4 RVQ codebooks, 50 Hz frame rate at 32kHz, 128-dim continuous embeddings pre-quantization
- Transformer hidden dim: 1024 (d_model)
- Vocab per codebook (card): 2048
- License: CC-BY-NC (research acceptable)

**Why not MusicGen-large:** Exceeds 4090 VRAM at full precision; LoRA feasibility uncertain.  
**Why not AudioLDM2:** CLAP compresses to single 512-d vector, discards temporal structure needed for phrase-level harmonic conditioning.  
**Why not Stable Audio Open:** No audio conditioning pathway.

### Conditioning Mechanism: AudioEmbeddingConditioner

`src/model/conditioning.py` — `AudioEmbeddingConditioner`

- Encodes GT audio via frozen EnCodec encoder (pre-quantization) → (B, D_enc=128, T_frames)
- Transposes → (B, T_frames, 128)
- Learned linear projection: Linear(128, 1024) + LayerNorm → (B, T_frames, 1024)
- Dropout(p=0.1) on conditioning — implements conditioning dropout for optional CFG at inference
- Passed as `cross_attention_src` to MusicGen transformer

**Why pre-quantization continuous embeddings (not discrete tokens):**
- Continuous embeddings retain soft probability mass across all codebook dimensions
- Post-quantization discrete tokens force nearest-neighbour assignment, collapsing polyphony
- For dense chords, pre-quant embeddings carry full simultaneous activation across codebook entries

**Why not CLAP:** Compresses full audio to single 512-d vector, losing temporal structure. 5s pattern → ~250 conditioning frames at 50Hz vs. 1 vector with CLAP.

**Critical engineering note:** EnCodec's SEANetEncoder contains an LSTM. LSTM CUDA kernel (`_thnn_fused_lstm_cell_cuda`) does NOT support bfloat16. All encoder calls use:
```python
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=False):
        enc = self.encoder(gt_audio.float())
```

### LoRA: Manual PyTorch Implementation

`src/model/lora_utils.py`

**Why not PEFT:** peft >= 0.8.0 requires `transformers >= 4.36` for the `Cache` class. audiocraft pins transformers to an older version. Three-way version conflict (torch / transformers / peft) has no solution. LoRA implemented from scratch in ~100 lines.

**Implementation:** Follows Hu et al. (2022) exactly: `h = W₀x + (BAx) × (α/r)`
- `W₀` frozen (requires_grad=False)
- `lora_A`: Kaiming uniform init
- `lora_B`: zero init (ensures adapter output = 0 at training start → stable fine-tuning)

**Hyperparameters:**
- rank = 32
- lora_alpha = 64 (= 2 × rank, effective scale = 2.0, constant across rank ablations)
- lora_dropout = 0.05

**Target layers:** `out_proj` only (96 layers, ~9.4M trainable params, 0.51% of total 1904.1M)

**Critical limitation:** audiocraft's `StreamingMultiheadAttention` uses fused `in_proj` (combined QKV) not separate `q_proj`/`k_proj`/`v_proj`. Discovery found only `out_proj`. Adding `in_proj` to `_ATTN_SUFFIXES` in `lora_utils.py` would also adapt the fused QKV projection — this is the ablation in Step 7.3.

**Parameter summary:**
- Total: 1904.1M
- Trainable: 9.6M (0.51%) — LoRA adapters + conditioner
- Frozen: 1894.5M

### VarianceEngineModel

`src/model/variance_engine.py` — `VarianceEngineModel(nn.Module)`

**Critical: MusicGen is NOT an nn.Module.** It is a high-level wrapper dataclass. `mg.parameters()` raises AttributeError. Must access `mg.lm` and `mg.compression_model` separately.

**d_model derivation:** `mg.lm.transformer.dim` does not exist. Use `mg.lm.linears[0].in_features` — output projection maps d_model → card, so in_features == d_model.

**d_encodec derivation:** No stable public attribute. `_infer_encodec_dim()` tries `dimension`, `output_dim`, `d_model` attributes, then falls back to dummy forward pass.

**Forward pass (training — teacher forcing):**
1. GT audio → conditioner → (B, T_gt_frames, 1024) cross-attention context
2. var_codes[:, :, :-1] → sum of per-codebook token embeddings → (B, T-1, 1024)
3. transformer(x, cross_attention_src=gt_cond) → (B, T-1, 1024)
4. linears[k](out) per codebook → stack → (B, n_q, T-1, card) logits
5. CE loss against var_codes[:, :, 1:] with attention mask

**Inference (generate):**
- Nucleus sampling (top-p=0.9)
- Temperature linearly spaced across [0.8, 1.2] for N variations
- Starts from BOS token (index = card = 2048)
- Autoregressive: generates up to max_new_tokens=750 frames (= 15s at 50Hz)

**Checkpoint save:** saves only LoRA weights (keys containing `lora_A` or `lora_B`) + conditioner state dict (~50 MB). Base MusicGen weights not saved (license compliance).

---

## Dataset Pipeline (Step 6a)

`src/data/dataset.py` — `VariancePairDataset`, `collate_fn`, `build_dataloaders`

**Audio loading:** `librosa.load(filepath, sr=None, mono=True)` — torchaudio.load failed (torchcodec backend, see Issues section).

**Cropping:**
- Train: random crop (augmentation — different phrase segments each epoch)
- Val/Test: centre crop (deterministic, reproducible metrics)
- GT and variation crops are NOT synchronised — different recordings, no temporal alignment

**Collate function:**
- Right-zero-pads GT and variation to longest in batch
- Attention mask: boolean tensor at EnCodec frame level (50 Hz), True = valid frame
- `_ENCODEC_HOP = 640` samples (32000 / 50)
- GT has no explicit mask — cross-attention learns to ignore zero-padded GT frames

**num_workers = 0:** EnCodec encoding requires CUDA; cannot be used in forked DataLoader workers.

---

## Training Loop (Step 6b)

`src/train.py`

Run: `python src/train.py --config configs/train_config.yaml`

**Key hyperparameters** (`configs/train_config.yaml`):

| Param | Value | Reason |
|---|---|---|
| batch_size | 4 | Per-GPU on RTX 4090 24GB |
| grad_accum_steps | 4 | Effective batch = 16 |
| lr_conditioner | 1e-3 | Randomly initialised; needs fast convergence |
| lr_lora | 1e-4 | Pre-trained weights; conservative update |
| β2 in AdamW | 0.95 | Faster adaptation for small-dataset fine-tuning (vs default 0.999) |
| weight_decay | 0.01 | Standard transformer fine-tuning |
| warmup_steps | 500 | ~0.35 epochs; stabilises conditioner before LR drops |
| precision | bf16 | RTX 4090 native bf16; no loss scaling needed |
| gradient_checkpointing | true | Reduces VRAM ~40% at ~30% compute cost |
| val_every_steps | 200 | ~7 checks/epoch; detects overfitting early |
| save_every_steps | 500 | Rolling window of 3 checkpoints + best.pt always kept |
| early_stopping_patience | 5 | Stop if val loss stagnates 5 consecutive evaluations |
| n_epochs | 100 | Upper bound; early stopping governs in practice |
| seed | 42 | Reproducibility |

**LR schedule:** Cosine decay with linear warmup, implemented as pure PyTorch `LambdaLR` (no transformers dependency — see Issues).

**Multi-GPU (DataParallel):**
- `model.transformer = nn.DataParallel(model.transformer)` when n_gpus > 1
- RTX 4090 (cuda:0) is primary; RTX 3090 (cuda:1) is secondary
- `compression_model` and `conditioner` stay on cuda:0 — called outside the DP forward
- All state dict access unwraps: `model.transformer.module if isinstance(...DataParallel) else model.transformer`

**Gradient clipping:** `clip_grad_norm_(all_params, max_norm=1.0)` before each optimizer step.

**EnCodec encode in training loop:** Wrapped in `autocast(enabled=False)` + `.float()` everywhere — bf16 LSTM issue.

**Checkpoint contents:**
```python
{
    "step": int,
    "val_loss": float,
    "conditioner_state": conditioner.state_dict(),
    "lora_state": {k: v for k, v in transformer.state_dict().items()
                   if "lora_A" in k or "lora_B" in k},
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
}
```

**Logging:** W&B (project="VarianceEngine"). Falls back to console if wandb not installed.

---

## Engineering Issues & Resolutions

All 9 issues encountered and resolved during development:

### Issue 1 — torchaudio torchcodec backend
**Symptom:** `torchaudio.load()` → `"System error"`. `torchaudio.save()` → same.  
**Cause:** torchaudio 2.3.0 defaulted to `torchcodec` backend; not functional on server.  
**Fix:** `librosa.load(sr=None, mono=True)` for loading; `soundfile.write(..., subtype="PCM_16")` for saving. `torchaudio.functional.resample` still used (pure tensor, no backend).  
**Files:** `src/preprocess.py`, `src/data/dataset.py`

### Issue 2 — transformers `register_pytree_node` AttributeError
**Symptom:** `from transformers import get_cosine_schedule_with_warmup` → `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`  
**Cause:** audiocraft pins `transformers==4.41.2`; this version's internal PyTorch API calls conflict with torch 2.3.  
**Fix:** Removed transformers import. Reimplemented `get_cosine_schedule_with_warmup` as pure PyTorch `LambdaLR` in `src/train.py`. Behaviour identical.  
**Files:** `src/train.py`

### Issue 3 — PEFT `Cache` ImportError
**Symptom:** `from peft import LoraConfig, get_peft_model` → `ImportError: cannot import name 'Cache' from 'transformers'`  
**Cause:** peft >= 0.8.0 requires `transformers >= 4.36` for `Cache` class. audiocraft's pinned transformers predates it. No three-way compatible version set exists.  
**Fix:** Removed PEFT entirely. Manual LoRA in `src/model/lora_utils.py` (~100 lines, follows Hu et al. 2022).  
**Files:** `src/model/lora_utils.py` (new), `src/model/variance_engine.py`, `requirements.txt`

### Issue 4 — MusicGen not an `nn.Module`
**Symptom:** `for p in mg.parameters()` → `AttributeError: 'MusicGen' object has no attribute 'parameters'`  
**Cause:** `MusicGen` is a high-level wrapper dataclass, not `nn.Module`.  
**Fix:** Access `mg.lm.parameters()` and `mg.compression_model.parameters()` separately.  
**Files:** `src/model/variance_engine.py`

### Issue 5 — `StreamingTransformer` has no `.dim`
**Symptom:** `mg.lm.transformer.dim` → `AttributeError`  
**Cause:** audiocraft's `StreamingTransformer` has no stable public `.dim` attribute.  
**Fix:** `d_model = mg.lm.linears[0].in_features` — output projection maps d_model → card, so in_features == d_model always.  
**Files:** `src/model/variance_engine.py`

### Issue 6 — EnCodec LSTM does not support bfloat16
**Symptom:** `RuntimeError: "thnn_fused_lstm_cell_cuda" not implemented for "BFloat16"`  
**Cause:** EnCodec `SEANetEncoder` contains bidirectional LSTM. PyTorch fused LSTM CUDA kernel has no bf16 support. Outer `autocast(bf16)` context cast encoder inputs to bf16.  
**Fix:** All encoder calls wrapped:
```python
with torch.cuda.amp.autocast(enabled=False):
    enc = encoder(audio.float())
```
Applied in `conditioning.py` forward, `variance_engine.py` encode helpers, `train.py` training loop and validation.  
**Files:** `src/model/conditioning.py`, `src/model/variance_engine.py`, `src/train.py`

### Issue 7 — LoRA layers on CPU, inputs on CUDA
**Symptom:** `RuntimeError: Expected all tensors to be on the same device, found cuda:0 and cpu`  
**Cause:** `nn.Linear(...)` in `LoRALinear.__init__` initialises on CPU by default. Wrapped base layer was on cuda:0.  
**Fix:**
```python
device = linear.weight.device
self.lora_A = nn.Linear(in_features, rank, bias=False).to(device)
self.lora_B = nn.Linear(rank, out_features, bias=False).to(device)
```
**Files:** `src/model/lora_utils.py`

### Issue 8 — DataParallel `.module` key prefix
**Symptom:** State dict keys prefixed `module.*` after DataParallel wrapping. Checkpoint save/load produced `KeyError`.  
**Cause:** `nn.DataParallel` wraps model as `.module`; all state dict keys acquire the prefix.  
**Fix:** Consistent unwrapping pattern everywhere:
```python
_t = model.transformer.module if isinstance(model.transformer, nn.DataParallel) else model.transformer
```
Applied in checkpoint save, checkpoint load, and optimizer param group construction.  
**Files:** `src/train.py`

### Issue 9 — LoRA only targets `out_proj` (not QKV)
**Symptom/Finding:** `get_lora_target_modules()` discovered only `out_proj`; 96 layers adapted instead of expected 384.  
**Cause:** audiocraft's `StreamingMultiheadAttention` uses fused `in_proj` (single linear for combined QKV), not separate `q_proj`/`k_proj`/`v_proj`.  
**Status:** Documented limitation. To adapt QKV: add `"in_proj"` to `_ATTN_SUFFIXES` in `src/model/lora_utils.py`. This is ablation Step 7.3.  
Current state: 96 `out_proj` layers, ~9.4M trainable params.

---

## File Map

```
VarianceEngine/
├── configs/
│   └── train_config.yaml          # All hyperparameters (single source of truth)
├── src/
│   ├── analyze_dataset.py         # Step 1: dataset audit, chroma, pYIN, plots
│   ├── preprocess.py              # Step 2: resample, normalize, split
│   ├── model/
│   │   ├── conditioning.py        # AudioEmbeddingConditioner (EnCodec→cross-attn ctx)
│   │   ├── lora_utils.py          # LoRALinear, apply_lora, count_parameters
│   │   └── variance_engine.py     # VarianceEngineModel: forward, compute_loss, generate
│   ├── data/
│   │   └── dataset.py             # VariancePairDataset, collate_fn, build_dataloaders
│   ├── train.py                   # Step 6: training loop, CheckpointManager
│   └── utils/
│       ├── audio.py               # load_audio, chroma_stft, pYIN, RMS, peak
│       └── io.py                  # parse_filename, collect_files, build_pairs
├── outputs/                       # Step 1+2 outputs (dataset_stats.json, splits.json, etc.)
├── preprocessed/                  # Step 2 output: cleaned 32kHz WAVs
├── checkpoints/                   # Step 6 output: step_*.pt, best.pt
├── PIPELINE.md                    # Research paper documentation (gitignored)
├── APPROACHES.md                  # Approach A–E evaluation (gitignored)
├── CLAUDE.md                      # Dataset context for Claude (gitignored)
├── MEMORY.md                      # This file
└── requirements.txt               # Pinned dependencies
```

---

## Key Dependencies & Version Constraints

```
torch==2.3.0
torchaudio==2.3.0
transformers==4.41.2          # pinned by audiocraft — do NOT upgrade
audiocraft==1.3.0
librosa==0.10.2
soundfile==0.12.1
numpy==1.26.4
fadtk==0.2.0                  # for FAD evaluation (Step 7)
wandb                         # optional, falls back to console logging
```

**Do not install peft.** Version conflict with audiocraft's transformers pin is unresolvable.  
**Do not upgrade transformers.** audiocraft will break.

---

## Current Status (as of 2026-05-21)

- Steps 1–6 complete and running on server
- Model training on RTX 4090 + RTX 3090
- All 9 engineering issues resolved
- PIPELINE.md fully documented including §6.5 Engineering Issues & Resolutions

**Remaining steps:**

### Step 7 — Evaluation (`src/evaluate.py`) — NOT YET WRITTEN
Metrics to implement:
- **FAD** (primary): VGGish embeddings via `fadtk`, compare generated vs. real variation distribution
- **MOS** (primary human): listening study after training completes — design TBD
- **CLAP similarity**: GT ↔ generated; LAION-CLAP audio encoder, cosine similarity
- **Chroma cosine similarity**: report full distribution, not just mean (heavily ornamental dataset)
- **Variation diversity**: mean pairwise CLAP distance across N generated outputs (detect collapse)

Ablations:
1. No GT conditioning (unconditional) — baseline
2. CLAP vector vs. EnCodec embedding conditioning — temporal resolution value
3. LoRA rank 16 vs. 32 vs. 64 — capacity across variation range
4. `in_proj` LoRA (QKV) vs. `out_proj` only — current limitation ablation
5. Artist ID auxiliary conditioning
6. Split by artist vs. by pattern — generalization type
7. Ornamental vs. structural variation subsets (chroma distance < 0.7 = structural; > 0.9 = ornamental)

### Step 8 — Inference script (`src/generate.py`) — NOT YET WRITTEN
Interface:
```bash
python src/generate.py --input pattern.wav --n_variations 5 --checkpoint checkpoints/best.pt
```
Output: `variation_1.wav` … `variation_N.wav`

Steps: load + resample to 32kHz → peak normalize → encode GT → condition transformer → nucleus sampling (top-p=0.9, T∈[0.8,1.2]) → decode → save.

### Step 9 — Reproducibility artifacts
- Pin final requirements.txt
- Release LoRA-only checkpoint (base MusicGen weights not redistributed — CC-BY-NC)
- Ensure splits.json committed for exact replication
- All eval outputs to `results/`

---

## Resuming Training

If training was interrupted, `CheckpointManager.load_latest()` resumes from last checkpoint:
```python
# Automatically called if checkpoint dir is non-empty on train.py startup
# NOT automatically implemented — add resume logic if needed:
step = ckpt_manager.load_latest(model, optimizer, scheduler, device)
```
Currently `train.py` does NOT auto-resume — it starts from scratch. To add resume: check if `checkpoints/` contains step_*.pt files at startup and call `load_latest` before the training loop.

**Best checkpoint:** `checkpoints/best.pt` — lowest val loss seen.

---

## Architectural Decisions Quick Reference

| Decision | Choice | Alternative rejected | Key reason |
|---|---|---|---|
| Foundation model | MusicGen-medium (1.5B) | MusicGen-large, AudioLDM2 | VRAM fit; temporal conditioning |
| Conditioning | EnCodec pre-quant embeddings | CLAP, post-quant tokens | Temporal resolution; polyphony retention |
| Fine-tuning | LoRA rank=32 on `out_proj` | Full fine-tune, PEFT | VRAM; catastrophic forgetting; PEFT version conflict |
| Audio I/O | librosa.load + soundfile.write | torchaudio | torchcodec backend failure |
| LR schedule | Cosine warmup (pure PyTorch) | transformers scheduler | transformers version conflict |
| Multi-GPU | DataParallel on transformer | DDP (torchrun) | Single-process simplicity; small trainable param count |
| Chroma algorithm | chroma_stft | chroma_cqt | Polyphony robustness; no bass ringing |
| Pitch estimation | pYIN | piptrack | Voiced/unvoiced model; no hallucinated candidates |
| Inference sampling | Nucleus (top-p=0.9) | Greedy, temperature-only | Diversity without degenerate tokens |
| Split strategy | By pattern_id | By file, by artist | Prevents data leakage across variation sets |
| Loss | Flat multi-codebook CE | Weighted by codebook rank | No validation signal at this data scale |
| β2 in AdamW | 0.95 | 0.999 (default) | Faster adaptation for small-dataset fine-tuning |
