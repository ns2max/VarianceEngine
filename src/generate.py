"""
generate.py — Step 8: Inference / Variation Generation

Project        : VarianceEngine
Pipeline stage : Step 8 of 9 (see PIPELINE.md)

Usage
-----
    # Generate 5 variations from a ground-truth WAV
    python src/generate.py \
        --input path/to/pattern.wav \
        --checkpoint checkpoints/best.pt \
        --n_variations 5 \
        --output_dir generated/

    # Override sampling parameters
    python src/generate.py \
        --input pattern.wav \
        --checkpoint checkpoints/best.pt \
        --n_variations 10 \
        --top_p 0.85 \
        --temperature_min 0.8 \
        --temperature_max 1.2 \
        --seed 42

Checkpoint format
-----------------
Accepts training checkpoints from train.py (keys: conditioner_state, lora_state)
and from VarianceEngineModel.save_trainable_weights() (keys: conditioner, lora).
Both formats are handled automatically.

Output
------
For --input pattern.wav --n_variations 5:
    generated/
        pattern_var_001.wav
        pattern_var_002.wav
        pattern_var_003.wav
        pattern_var_004.wav
        pattern_var_005.wav
        generation_info.json   — metadata: checkpoint, config, val_loss, step
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.variance_engine import VarianceEngineModel


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

TARGET_SR = 32_000
PEAK_DBFS = -1.0


def load_and_prepare(path: str) -> torch.Tensor:
    """Load WAV, resample to 32kHz, peak-normalise. Returns (1, 1, T) tensor."""
    audio, sr = librosa.load(path, sr=None, mono=True)
    waveform = torch.from_numpy(audio)  # (T,)

    if sr != TARGET_SR:
        import torchaudio.functional as AF
        waveform = AF.resample(waveform.unsqueeze(0), sr, TARGET_SR).squeeze(0)

    # Peak normalise to -1 dBFS
    peak = waveform.abs().max()
    if peak > 1e-8:
        target = 10 ** (PEAK_DBFS / 20)
        waveform = waveform * (target / peak)

    return waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)


def save_wav(waveform: torch.Tensor, path: Path) -> None:
    """Save (1, 1, T) or (1, T) float32 tensor as 16-bit PCM WAV."""
    audio = waveform.squeeze().cpu().numpy()
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
    model: VarianceEngineModel,
    device: str,
) -> dict:
    """Load LoRA + conditioner weights into model from a training checkpoint.

    Handles two checkpoint formats:
      - train.py format:  keys = conditioner_state, lora_state, step, val_loss
      - save_trainable_weights format: keys = conditioner, lora, config

    Also strips DataParallel 'module.' prefix from lora keys if present
    (checkpoints saved during multi-GPU training have this prefix).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Detect format
    if "conditioner_state" in ckpt:
        conditioner_state = ckpt["conditioner_state"]
        lora_state = ckpt["lora_state"]
        meta = {"step": ckpt.get("step"), "val_loss": ckpt.get("val_loss")}
    elif "conditioner" in ckpt:
        conditioner_state = ckpt["conditioner"]
        lora_state = ckpt["lora"]
        meta = {}
    else:
        raise ValueError(
            f"Unrecognised checkpoint format. Keys found: {list(ckpt.keys())}"
        )

    # Strip DataParallel 'module.' prefix if present
    lora_state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in lora_state.items()
    }

    model.conditioner.load_state_dict(conditioner_state, strict=True)
    missing, unexpected = model.transformer.load_state_dict(lora_state, strict=False)
    if unexpected:
        print(f"  Warning: unexpected checkpoint keys: {unexpected[:3]}")

    n_lora = len(lora_state)
    print(f"  Loaded {n_lora} LoRA weight tensors")
    if meta.get("step"):
        print(f"  Checkpoint: step={meta['step']}, val_loss={meta['val_loss']:.4f}")

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stem for output filenames
    input_stem = Path(args.input).stem

    # ------------------------------------------------------------------
    # Load and prepare GT audio
    # ------------------------------------------------------------------
    print(f"\nLoading GT audio: {args.input}")
    gt_audio = load_and_prepare(args.input).to(device)
    duration_s = gt_audio.shape[-1] / TARGET_SR
    print(f"  Duration: {duration_s:.2f}s  Shape: {gt_audio.shape}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print(f"\nLoading VarianceEngineModel ({args.musicgen_name})...")
    model = VarianceEngineModel(
        musicgen_name=args.musicgen_name,
        lora_rank=args.lora_rank,
        device=device,
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    meta = load_checkpoint(Path(args.checkpoint), model, device)
    model.eval()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    # max_new_tokens: generate same length as input (capped at 750 frames = 15s)
    input_frames = gt_audio.shape[-1] // 640  # _ENCODEC_HOP = 640
    max_new_tokens = min(input_frames + 50, 750)  # slight headroom over input length

    print(f"\nGenerating {args.n_variations} variations...")
    print(f"  top_p={args.top_p}  T=[{args.temperature_min}, {args.temperature_max}]"
          f"  max_tokens={max_new_tokens}  seed={args.seed}")

    waveforms = model.generate(
        gt_audio=gt_audio,
        n_variations=args.n_variations,
        max_new_tokens=max_new_tokens,
        top_p=args.top_p,
        temperature_range=(args.temperature_min, args.temperature_max),
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"\nSaving to {output_dir}/")
    output_paths = []
    for i, wav in enumerate(waveforms, start=1):
        out_path = output_dir / f"{input_stem}_var_{i:03d}.wav"
        save_wav(wav, out_path)
        dur = wav.shape[-1] / TARGET_SR
        print(f"  [{i}/{args.n_variations}] {out_path.name}  ({dur:.2f}s)")
        output_paths.append(str(out_path))

    # Save generation metadata
    info = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "musicgen_name": args.musicgen_name,
        "lora_rank": args.lora_rank,
        "n_variations": args.n_variations,
        "top_p": args.top_p,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "seed": args.seed,
        "max_new_tokens": max_new_tokens,
        "input_duration_s": round(duration_s, 3),
        "checkpoint_step": meta.get("step"),
        "checkpoint_val_loss": meta.get("val_loss"),
        "outputs": output_paths,
    }
    info_path = output_dir / "generation_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nMetadata → {info_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VarianceEngine — Generate N variations from a ground-truth WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to ground-truth WAV file (any sample rate; resampled internally to 32kHz).",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt",
        help="Path to training checkpoint (.pt file from train.py or save_trainable_weights).",
    )
    parser.add_argument(
        "--n_variations", type=int, default=5,
        help="Number of variations to generate.",
    )
    parser.add_argument(
        "--output_dir", default="generated/",
        help="Directory to save generated WAVs and generation_info.json.",
    )
    parser.add_argument(
        "--musicgen_name", default="facebook/musicgen-medium",
        help="MusicGen model ID (must match the checkpoint).",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=32,
        help="LoRA rank (must match the checkpoint).",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus sampling probability mass threshold.",
    )
    parser.add_argument(
        "--temperature_min", type=float, default=0.8,
        help="Minimum sampling temperature (applied to variation 1).",
    )
    parser.add_argument(
        "--temperature_max", type=float, default=1.2,
        help="Maximum sampling temperature (applied to variation N).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed. Variation i uses seed+i.",
    )

    args = parser.parse_args()
    generate(args)
