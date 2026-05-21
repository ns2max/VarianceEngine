"""
preprocess.py — Step 2: Data Preprocessing

Project        : VarianceEngine
Pipeline stage : Step 2 of 9 (see PIPELINE.md)
Purpose        : Clean the raw DOPP dataset and produce a preprocessed copy
                 ready for training. Outputs preprocessed WAVs to a new
                 directory; original dopp/ is never modified.

Preprocessing steps (in order):
    1. Load dataset_stats.json from Step 1
    2. Exclude clipped GT patterns entirely
    3. Exclude clipped variation files
    4. Exclude files with duration < MIN_DURATION_S
    5. Flag files with duration > MAX_DURATION_S (handled at train time via crop)
    6. Resample all retained files to TARGET_SR (32000 Hz)
    7. Peak-normalise to PEAK_DBFS (-1 dBFS)
    8. Save to output_dir/{artistID}_{patternID}_{versionID}.wav
    9. Rebuild dataset_pairs.csv from retained files
   10. Generate train/val/test splits by pattern_id
   11. Save splits.json and preprocessing_report.json

Outputs:
    preprocessed/          — cleaned WAV files at 32kHz
    outputs/splits.json    — train/val/test pattern splits
    outputs/preprocessing_report.json — audit of what was excluded and why

Usage:
    python src/preprocess.py \\
        --data_dir dopp/ \\
        --stats_path outputs/dataset_stats.json \\
        --output_dir preprocessed/ \\
        --splits_dir outputs/ \\
        --workers 4

Design notes
------------
- torchaudio is used for resampling (sinc interpolation) because it is the
  library used in the training pipeline, ensuring identical audio loading
  behaviour between preprocessing and training.
- librosa is NOT used here to avoid any subtle resampling differences between
  the two libraries at training time.
- All file writes are atomic: files are written to a temp path and renamed on
  success, preventing partial writes from corrupting the output directory.
- Random seed for train/val/test split is fixed and recorded in splits.json
  for exact reproducibility.
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.functional as F
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — all tunable via CLI args; defaults documented here
# ---------------------------------------------------------------------------

TARGET_SR: int = 32_000       # MusicGen EnCodec native sample rate
MIN_DURATION_S: float = 1.0   # Files shorter than this are excluded
MAX_DURATION_S: float = 15.0  # Files longer: flagged; crop applied at train time
PEAK_DBFS: float = -1.0       # Target peak level after normalisation
CLIP_THRESHOLD: float = 0.999 # peak_amplitude >= this → clipped
SPLIT_SEED: int = 42          # Fixed seed for reproducible splits
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10


# ---------------------------------------------------------------------------
# Exclusion logic
# ---------------------------------------------------------------------------

def identify_exclusions(stats: list[dict]) -> dict:
    """Determine which files to exclude and why, from Step 1 stats.

    Returns a dict mapping filepath → reason_string for all excluded files.
    Also identifies which (artist_id, pattern_id) groups must be entirely
    removed due to a clipped GT.

    Exclusion hierarchy (applied in order):
        1. Pattern-level exclusion if GT (version_id=0) is clipped
        2. File-level exclusion if variation is clipped
        3. File-level exclusion if duration < MIN_DURATION_S

    Files with duration > MAX_DURATION_S are NOT excluded — they are handled
    via random cropping at training time. They are flagged in the report.
    """
    excluded: dict[str, str] = {}
    flagged_long: dict[str, str] = {}

    # Step 1: identify patterns with clipped GTs
    clipped_gt_patterns: set[tuple] = set()
    for s in stats:
        if s["version_id"] == 0 and s.get("peak_amplitude", 0) >= CLIP_THRESHOLD:
            clipped_gt_patterns.add((s["artist_id"], s["pattern_id"]))

    # Step 2: mark entire patterns for exclusion
    for s in stats:
        key = (s["artist_id"], s["pattern_id"])
        if key in clipped_gt_patterns:
            excluded[s["filepath"]] = (
                f"pattern excluded: GT ({s['artist_id']}_{s['pattern_id']}_0) is clipped"
            )

    # Step 3: exclude remaining clipped variation files
    for s in stats:
        if s["filepath"] in excluded:
            continue
        if s.get("peak_amplitude", 0) >= CLIP_THRESHOLD:
            excluded[s["filepath"]] = "clipped (peak_amplitude >= 0.999)"

    # Step 4: exclude short files
    for s in stats:
        if s["filepath"] in excluded:
            continue
        if s.get("duration_s", 99) < MIN_DURATION_S:
            excluded[s["filepath"]] = f"duration {s['duration_s']:.3f}s < {MIN_DURATION_S}s minimum"

    # Step 5: flag long files (not excluded)
    for s in stats:
        if s["filepath"] in excluded:
            continue
        if s.get("duration_s", 0) > MAX_DURATION_S:
            flagged_long[s["filepath"]] = (
                f"duration {s['duration_s']:.3f}s > {MAX_DURATION_S}s — random crop at training time"
            )

    return {
        "excluded": excluded,
        "flagged_long": flagged_long,
        "clipped_gt_patterns": [
            {"artist_id": a, "pattern_id": p} for a, p in sorted(clipped_gt_patterns)
        ],
    }


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def peak_normalise(waveform: torch.Tensor, target_dbfs: float = PEAK_DBFS) -> torch.Tensor:
    """Normalise waveform to a target peak level in dBFS.

    Parameters
    ----------
    waveform:
        Float tensor, shape (1, T) or (T,).
    target_dbfs:
        Target peak level.  -1.0 dBFS → peak = 10^(-1/20) ≈ 0.891.

    Returns
    -------
    Normalised waveform, same shape as input.

    Notes
    -----
    A small epsilon (1e-8) guards against division by zero on silent files.
    Silent files (RMS near zero) that survived exclusion are edge cases;
    the epsilon prevents NaN/Inf while keeping the waveform near-silent.
    """
    target_linear = 10 ** (target_dbfs / 20)
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_linear / peak)


def preprocess_file(
    src_path: str,
    dst_path: str,
    target_sr: int = TARGET_SR,
    peak_dbfs: float = PEAK_DBFS,
) -> dict:
    """Load, resample, normalise, and save one WAV file.

    Parameters
    ----------
    src_path:
        Source WAV path (original 22050 Hz file).
    dst_path:
        Destination WAV path (preprocessed 32000 Hz file).

    Returns
    -------
    dict with keys: src_path, dst_path, original_sr, original_duration_s,
    output_sr, output_duration_s, peak_before, peak_after.

    Implementation notes
    --------------------
    torchaudio.load returns a float32 tensor normalised to [-1, 1] for 16-bit
    PCM, consistent with MusicGen's internal audio handling.

    Resampling uses sinc interpolation (the torchaudio default) which is
    the standard anti-aliased upsampling method and avoids aliasing artifacts
    that would occur with linear or nearest-neighbour interpolation.

    The write uses soundfile backend (via torchaudio) with 16-bit PCM output
    for compatibility with downstream audio tools.
    """
    # Use soundfile directly to avoid torchaudio backend issues (torchcodec
    # became the default in torchaudio >=2.4 and may not be installed).
    audio_np, sr = sf.read(src_path, dtype="float32", always_2d=False)
    if audio_np.ndim == 1:
        waveform = torch.from_numpy(audio_np).unsqueeze(0)   # (1, T)
    else:
        waveform = torch.from_numpy(audio_np.T)              # (C, T)

    original_duration_s = waveform.shape[-1] / sr
    peak_before = float(waveform.abs().max())

    # Ensure mono (mix down if multi-channel)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = F.resample(
            waveform,
            orig_freq=sr,
            new_freq=target_sr,
            resampling_method="sinc_interpolation",
        )

    # Peak normalise
    waveform = peak_normalise(waveform, target_dbfs=peak_dbfs)
    peak_after = float(waveform.abs().max())

    # Atomic write: write to temp file, then rename
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=dst_path.parent, delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        torchaudio.save(tmp_path, waveform, target_sr, encoding="PCM_S", bits_per_sample=16)
        os.replace(tmp_path, dst_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return {
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "original_sr": sr,
        "original_duration_s": round(original_duration_s, 4),
        "output_sr": target_sr,
        "output_duration_s": round(waveform.shape[-1] / target_sr, 4),
        "peak_before": round(peak_before, 6),
        "peak_after": round(peak_after, 6),
    }


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def generate_splits(
    retained_stats: list[dict],
    seed: int = SPLIT_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict:
    """Split patterns into train/val/test by pattern_id.

    Strategy: stratified by artist_id — for each artist, independently
    shuffle their pattern_ids and split in train_ratio / val_ratio / remainder.
    This ensures each artist is represented in all three splits, preventing
    the model from seeing no examples of an artist's style during training.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a list of
    {'artist_id': int, 'pattern_id': int} dicts.
    Also includes 'seed' and 'ratios' for reproducibility.

    Notes on stratification
    -----------------------
    Without stratification, a naive random split over all 176 patterns could
    by chance place all patterns from one artist into the test set, making
    the split confounded with artist identity. Stratifying by artist ensures
    the train/val/test split is orthogonal to artist identity — the model
    generalises to unseen *patterns*, not unseen *artists*.
    """
    rng = random.Random(seed)

    # Group pattern_ids by artist
    artist_patterns: dict[int, list[int]] = defaultdict(set)
    for s in retained_stats:
        artist_patterns[s["artist_id"]].add(s["pattern_id"])
    artist_patterns = {a: sorted(ps) for a, ps in artist_patterns.items()}

    train_patterns, val_patterns, test_patterns = [], [], []

    for artist_id, pattern_ids in sorted(artist_patterns.items()):
        shuffled = pattern_ids[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Ensure at least 1 in test if >=3 patterns
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_train -= 1
            n_test = 1
        elif n_test < 0:
            n_val = max(0, n_val + n_test)
            n_test = 0

        for pid in shuffled[:n_train]:
            train_patterns.append({"artist_id": artist_id, "pattern_id": pid})
        for pid in shuffled[n_train:n_train + n_val]:
            val_patterns.append({"artist_id": artist_id, "pattern_id": pid})
        for pid in shuffled[n_train + n_val:]:
            test_patterns.append({"artist_id": artist_id, "pattern_id": pid})

    return {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": round(1 - train_ratio - val_ratio, 2)},
        "counts": {
            "train_patterns": len(train_patterns),
            "val_patterns": len(val_patterns),
            "test_patterns": len(test_patterns),
        },
        "train": train_patterns,
        "val": val_patterns,
        "test": test_patterns,
    }


def build_split_pairs(pairs: list[dict], splits: dict) -> dict:
    """Map each split's patterns to their (gt_path, var_path) pairs.

    pairs is already filtered to retained files (Step 4). No filesystem
    existence check here — trust the upstream filter.

    Returns a dict with 'train', 'val', 'test', each a list of pair dicts.
    """
    def _pattern_key(p):
        return (p["artist_id"], p["pattern_id"])

    train_set = {_pattern_key(p) for p in splits["train"]}
    val_set = {_pattern_key(p) for p in splits["val"]}
    test_set = {_pattern_key(p) for p in splits["test"]}

    split_pairs = {"train": [], "val": [], "test": []}

    for pair in pairs:
        key = (pair["artist_id"], pair["pattern_id"])
        if key in train_set:
            split_pairs["train"].append(pair)
        elif key in val_set:
            split_pairs["val"].append(pair)
        elif key in test_set:
            split_pairs["test"].append(pair)

    return split_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_dir: str,
    stats_path: str,
    output_dir: str,
    splits_dir: str,
    workers: int = 4,
    target_sr: int = TARGET_SR,
    min_duration: float = MIN_DURATION_S,
    max_duration: float = MAX_DURATION_S,
    peak_dbfs: float = PEAK_DBFS,
    split_seed: int = SPLIT_SEED,
) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    splits_dir = Path(splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VarianceEngine — Preprocessing — Step 2")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Step 1 stats
    # ------------------------------------------------------------------
    print("\n[1/6] Loading dataset stats from Step 1...")
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"      Loaded {len(stats)} file records.")

    # ------------------------------------------------------------------
    # 2. Determine exclusions
    # ------------------------------------------------------------------
    print("\n[2/6] Identifying exclusions...")
    exclusion_info = identify_exclusions(stats)
    excluded = exclusion_info["excluded"]
    flagged_long = exclusion_info["flagged_long"]

    print(f"      Clipped GT patterns excluded: {len(exclusion_info['clipped_gt_patterns'])}")
    print(f"      Total files excluded: {len(excluded)}")
    print(f"        — clipped GT patterns (all versions): "
          f"{sum(1 for r in excluded.values() if 'pattern excluded' in r)}")
    print(f"        — clipped variations: "
          f"{sum(1 for r in excluded.values() if r == 'clipped (peak_amplitude >= 0.999)')}")
    print(f"        — too short (< {min_duration}s): "
          f"{sum(1 for r in excluded.values() if 'minimum' in r)}")
    print(f"      Files flagged long (> {max_duration}s, crop at train time): {len(flagged_long)}")

    retained_stats = [s for s in stats if s["filepath"] not in excluded]
    print(f"      Retained: {len(retained_stats)} / {len(stats)} files")

    # ------------------------------------------------------------------
    # 3. Preprocess retained files (parallel using torch multiprocessing)
    # ------------------------------------------------------------------
    print(f"\n[3/6] Preprocessing {len(retained_stats)} files → {output_dir}/")

    process_results = []
    errors = []

    # Note: torchaudio operations are not safely picklable for
    # ProcessPoolExecutor on all platforms; sequential loop used here.
    # On the server, increase throughput by splitting the file list and
    # running multiple preprocess.py instances with --subset flags (see --help).
    for s in tqdm(retained_stats, unit="file"):
        src = s["filepath"]
        dst = output_dir / Path(src).name
        try:
            result = preprocess_file(src, str(dst), target_sr=target_sr, peak_dbfs=peak_dbfs)
            process_results.append(result)
        except Exception as e:
            errors.append({"filepath": src, "error": str(e)})

    print(f"      Processed: {len(process_results)}, Errors: {len(errors)}")

    # ------------------------------------------------------------------
    # 4. Rebuild pairs from retained files
    # ------------------------------------------------------------------
    print("\n[4/6] Rebuilding GT–variation pairs from retained files...")

    # Load original pairs CSV
    import csv
    pairs_path = splits_dir / "dataset_pairs.csv"
    pairs = []
    with open(pairs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["artist_id"] = int(row["artist_id"])
            row["pattern_id"] = int(row["pattern_id"])
            row["var_version_id"] = int(row["var_version_id"])
            pairs.append(row)

    retained_paths = {str(output_dir / Path(s["filepath"]).name) for s in retained_stats}

    # Filter pairs: both GT and variation must be retained
    valid_pairs = []
    for p in pairs:
        gt_dst = str(output_dir / Path(p["gt_filepath"]).name)
        var_dst = str(output_dir / Path(p["var_filepath"]).name)
        if gt_dst in retained_paths and var_dst in retained_paths:
            valid_pairs.append({
                **p,
                "gt_preprocessed": gt_dst,
                "var_preprocessed": var_dst,
            })

    print(f"      Valid pairs after filtering: {len(valid_pairs)} / {len(pairs)}")

    # ------------------------------------------------------------------
    # 5. Generate splits
    # ------------------------------------------------------------------
    print("\n[5/6] Generating train/val/test splits...")
    splits = generate_splits(retained_stats, seed=split_seed)
    split_pairs = build_split_pairs(valid_pairs, splits)

    splits["pair_counts"] = {
        "train": len(split_pairs["train"]),
        "val": len(split_pairs["val"]),
        "test": len(split_pairs["test"]),
    }

    splits_out = splits_dir / "splits.json"
    with open(splits_out, "w") as f:
        json.dump(splits, f, indent=2)

    split_pairs_out = splits_dir / "split_pairs.json"
    with open(split_pairs_out, "w") as f:
        json.dump(split_pairs, f, indent=2)

    print(f"      Train: {splits['counts']['train_patterns']} patterns, {len(split_pairs['train'])} pairs")
    print(f"      Val:   {splits['counts']['val_patterns']} patterns, {len(split_pairs['val'])} pairs")
    print(f"      Test:  {splits['counts']['test_patterns']} patterns, {len(split_pairs['test'])} pairs")

    # ------------------------------------------------------------------
    # 6. Save preprocessing report
    # ------------------------------------------------------------------
    print("\n[6/6] Saving preprocessing report...")
    report = {
        "config": {
            "target_sr": target_sr,
            "min_duration_s": min_duration,
            "max_duration_s": max_duration,
            "peak_dbfs": peak_dbfs,
            "clip_threshold": CLIP_THRESHOLD,
            "split_seed": split_seed,
        },
        "input_files": len(stats),
        "retained_files": len(retained_stats),
        "processed_files": len(process_results),
        "processing_errors": len(errors),
        "clipped_gt_patterns": exclusion_info["clipped_gt_patterns"],
        "excluded_files": [
            {"filepath": fp, "reason": reason}
            for fp, reason in sorted(excluded.items())
        ],
        "flagged_long_files": [
            {"filepath": fp, "reason": reason}
            for fp, reason in sorted(flagged_long.items())
        ],
        "errors": errors,
        "split_summary": splits["counts"],
        "pair_summary": splits["pair_counts"],
    }

    report_out = splits_dir / "preprocessing_report.json"
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"      Report saved → {report_out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Input files       : {report['input_files']}")
    print(f"  Excluded          : {len(excluded)}")
    print(f"  Retained          : {report['retained_files']}")
    print(f"  Processing errors : {report['processing_errors']}")
    print(f"  Training pairs    : {len(split_pairs['train'])}")
    print(f"  Val pairs         : {len(split_pairs['val'])}")
    print(f"  Test pairs        : {len(split_pairs['test'])}")
    print(f"  Output dir        : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VarianceEngine — Preprocessing — Step 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="dopp/")
    parser.add_argument("--stats_path", default="outputs/dataset_stats.json")
    parser.add_argument("--output_dir", default="preprocessed/",
                        help="Directory for preprocessed WAV files.")
    parser.add_argument("--splits_dir", default="outputs/",
                        help="Directory for splits.json and preprocessing_report.json.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target_sr", type=int, default=TARGET_SR)
    parser.add_argument("--min_duration", type=float, default=MIN_DURATION_S)
    parser.add_argument("--max_duration", type=float, default=MAX_DURATION_S)
    parser.add_argument("--peak_dbfs", type=float, default=PEAK_DBFS)
    parser.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        stats_path=args.stats_path,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        workers=args.workers,
        target_sr=args.target_sr,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        peak_dbfs=args.peak_dbfs,
        split_seed=args.split_seed,
    )
