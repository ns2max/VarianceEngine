"""
analyze_dataset.py — Step 1: Dataset Analysis & Audit

Project        : VarianceEngine
Pipeline stage : Step 1 of 9 (see PIPELINE.md)
Purpose        : Characterise the DOPP dataset before any modelling work.
Outputs        : outputs/dataset_stats.json
                 outputs/dataset_pairs.csv
                 outputs/chroma_analysis.json
                 outputs/figures/duration_distribution.png
                 outputs/figures/version_count_distribution.png
                 outputs/figures/pitch_range_distribution.png
                 outputs/figures/chroma_similarity_gt_vs_variations.png

Usage
-----
    python src/analyze_dataset.py --data_dir dopp/ --output_dir outputs/

Design rationale
----------------
This script is intentionally separated from the training pipeline so that
dataset characterisation can be re-run cheaply and independently of model
changes.  All expensive computations (chroma, pYIN) are done once and cached
in JSON; downstream scripts read those files rather than re-processing audio.

Parallelism
-----------
Per-file feature extraction is embarrassingly parallel.  ProcessPoolExecutor
is used with workers = min(CPU_count, 8) to avoid memory pressure when many
large audio files are loaded simultaneously.  Each worker loads and processes
one file independently — no shared state.

Reproducibility
---------------
No random operations in this script.  Given the same input files, outputs
are deterministic.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for server use
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Project utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.io import collect_files, build_pairs, save_csv, save_json
from src.utils.audio import (
    load_audio,
    peak_amplitude,
    rms_amplitude,
    duration_seconds,
    compute_chroma,
    chroma_mean,
    chroma_cosine_similarity,
    estimate_pitch_range,
)


# ---------------------------------------------------------------------------
# Per-file worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_file(record: dict) -> dict:
    """Extract all per-file statistics for one WAV.

    Runs inside a worker process.  Must be a module-level function for
    ProcessPoolExecutor pickling to work on all platforms.

    Returns the input record dict augmented with computed statistics, or
    augmented with an 'error' key if loading fails.
    """
    try:
        waveform, sr = load_audio(record["filepath"])

        peak = peak_amplitude(waveform)
        rms = rms_amplitude(waveform)
        dur = duration_seconds(waveform, sr)

        chroma = compute_chroma(waveform, sr)
        chroma_centroid = chroma_mean(chroma).tolist()  # JSON-serialisable

        pitch_info = estimate_pitch_range(waveform, sr)

        return {
            **record,
            "sample_rate": sr,
            "duration_s": round(dur, 4),
            "peak_amplitude": round(peak, 6),
            "rms_amplitude": round(rms, 6),
            "chroma_centroid": chroma_centroid,
            **{f"pitch_{k}": v for k, v in pitch_info.items()},
        }

    except Exception as exc:
        return {**record, "error": str(exc)}


# ---------------------------------------------------------------------------
# Pattern-level chroma analysis
# ---------------------------------------------------------------------------

def compute_pattern_chroma_stats(stats: list[dict]) -> dict:
    """Aggregate per-file chroma data into per-pattern summaries.

    For each (artist_id, pattern_id) group:
    - GT chroma centroid (version_id=0)
    - Mean chroma similarity of all variations to the GT
    - Std of chroma similarities (spread of harmonic variation)
    - Min and max similarity (bounds of harmonic divergence)

    The chroma similarity distribution is the key characterisation for the
    paper: a bimodal distribution (cluster near 1.0 for ornamental variations,
    cluster at lower values for structural transpositions) would confirm that
    the dataset contains both variation magnitudes as stated.

    Returns
    -------
    dict keyed by "{artist_id}_{pattern_id}" with the summary stats.
    """
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for s in stats:
        if "error" in s or s.get("chroma_centroid") is None:
            continue
        key = f"{s['artist_id']}_{s['pattern_id']}"
        groups[key].append(s)

    pattern_stats = {}
    for key, group in groups.items():
        gt = next((s for s in group if s["version_id"] == 0), None)
        variations = [s for s in group if s["version_id"] > 0]

        if gt is None or not variations:
            continue

        gt_chroma = np.array(gt["chroma_centroid"])
        sims = [
            chroma_cosine_similarity(gt_chroma, np.array(v["chroma_centroid"]))
            for v in variations
        ]

        pattern_stats[key] = {
            "artist_id": gt["artist_id"],
            "pattern_id": gt["pattern_id"],
            "n_variations": len(variations),
            "gt_duration_s": gt["duration_s"],
            "gt_chroma_centroid": gt["chroma_centroid"],
            "chroma_sim_mean": round(float(np.mean(sims)), 4),
            "chroma_sim_std": round(float(np.std(sims)), 4),
            "chroma_sim_min": round(float(np.min(sims)), 4),
            "chroma_sim_max": round(float(np.max(sims)), 4),
            "chroma_similarities": [round(s, 4) for s in sims],
        }

    return pattern_stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path}")


def plot_duration_distribution(stats: list[dict], output_dir: Path) -> None:
    """Histogram of per-file durations, split by version_id=0 vs. variations.

    Separating GT from variations reveals whether ground truth patterns have
    systematically different lengths from their variations — a potential
    confound if the model uses duration as an implicit cue.
    """
    durations_gt = [s["duration_s"] for s in stats if s.get("version_id") == 0 and "error" not in s]
    durations_var = [s["duration_s"] for s in stats if s.get("version_id", 0) > 0 and "error" not in s]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durations_gt, bins=30, alpha=0.7, label=f"Ground truth (n={len(durations_gt)})", color="#2196F3")
    ax.hist(durations_var, bins=30, alpha=0.7, label=f"Variations (n={len(durations_var)})", color="#FF5722")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Duration Distribution — Ground Truth vs. Variations")
    ax.legend()
    sns.despine(ax=ax)
    _save_fig(fig, output_dir / "duration_distribution.png")


def plot_version_count_distribution(stats: list[dict], output_dir: Path) -> None:
    """Bar chart of how many versions each pattern has.

    Highly variable version counts could bias training toward patterns with
    many variations.  This plot informs whether weighted sampling is needed
    in the DataLoader (Step 6).
    """
    from collections import Counter
    counts = Counter(
        f"{s['artist_id']}_{s['pattern_id']}"
        for s in stats
        if "error" not in s and s.get("version_id", 0) > 0
    )
    version_counts = sorted(counts.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(version_counts, bins=range(min(version_counts), max(version_counts) + 2), align="left")
    ax.set_xlabel("Number of variations per pattern")
    ax.set_ylabel("Number of patterns")
    ax.set_title("Variation Count Distribution per Pattern")
    sns.despine(ax=ax)
    _save_fig(fig, output_dir / "version_count_distribution.png")


def plot_pitch_range_distribution(stats: list[dict], output_dir: Path) -> None:
    """Histogram of per-file pitch range (in semitones) from pYIN estimation.

    Patterns with a wide pitch range (>24 st = 2 octaves) are likely to
    contain structural transpositions or wide melodic contours.  This informs
    the evaluation: the chroma similarity threshold for 'ornamental' vs.
    'structural' variation subsets (Step 7.3 ablation).
    """
    ranges = [
        s["pitch_f0_range_st"]
        for s in stats
        if "error" not in s and s.get("pitch_f0_range_st") is not None
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ranges, bins=30, color="#4CAF50")
    ax.axvline(12, color="red", linestyle="--", alpha=0.6, label="1 octave")
    ax.axvline(24, color="orange", linestyle="--", alpha=0.6, label="2 octaves")
    ax.set_xlabel("Pitch range (semitones)")
    ax.set_ylabel("Count")
    ax.set_title("Pitch Range Distribution (pYIN dominant f0)")
    ax.legend()
    sns.despine(ax=ax)
    _save_fig(fig, output_dir / "pitch_range_distribution.png")


def plot_chroma_similarity_distribution(pattern_stats: dict, output_dir: Path) -> None:
    """Distribution of chroma cosine similarities (GT vs. each variation).

    This is the most important diagnostic plot for the paper.  A bimodal
    distribution would confirm the dataset contains both ornamental (high
    similarity) and structural (low similarity) harmonic variations.  A
    unimodal distribution near 1.0 would suggest mostly ornamental variation
    and would inform evaluation threshold choices.
    """
    all_sims = []
    for p in pattern_stats.values():
        all_sims.extend(p["chroma_similarities"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall distribution
    axes[0].hist(all_sims, bins=50, color="#9C27B0", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Chroma cosine similarity (GT vs. variation)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Harmonic Similarity: GT vs. All Variations (n={len(all_sims)})")
    axes[0].axvline(np.median(all_sims), color="red", linestyle="--", label=f"Median={np.median(all_sims):.3f}")
    axes[0].legend()
    sns.despine(ax=axes[0])

    # Per-pattern mean similarity vs. std (scatter — reveals which patterns have wide harmonic spread)
    means = [p["chroma_sim_mean"] for p in pattern_stats.values()]
    stds = [p["chroma_sim_std"] for p in pattern_stats.values()]
    axes[1].scatter(means, stds, alpha=0.6, s=20, color="#FF9800")
    axes[1].set_xlabel("Mean chroma similarity (per pattern)")
    axes[1].set_ylabel("Std of chroma similarity (per pattern)")
    axes[1].set_title("Per-Pattern Harmonic Variation Spread")
    sns.despine(ax=axes[1])

    fig.tight_layout()
    _save_fig(fig, output_dir / "chroma_similarity_gt_vs_variations.png")


# ---------------------------------------------------------------------------
# Summary statistics (printed + saved)
# ---------------------------------------------------------------------------

def compute_summary(stats: list[dict], pattern_stats: dict) -> dict:
    """Compute dataset-level summary statistics for the paper's data section."""
    valid = [s for s in stats if "error" not in s]
    errors = [s for s in stats if "error" in s]

    durations = [s["duration_s"] for s in valid]
    peaks = [s["peak_amplitude"] for s in valid]
    rmss = [s["rms_amplitude"] for s in valid]

    all_sims = []
    for p in pattern_stats.values():
        all_sims.extend(p["chroma_similarities"])

    version_counts = {}
    for s in valid:
        k = f"{s['artist_id']}_{s['pattern_id']}"
        version_counts[k] = version_counts.get(k, 0) + (1 if s["version_id"] > 0 else 0)
    vc = list(version_counts.values())

    return {
        "total_files": len(stats),
        "valid_files": len(valid),
        "error_files": len(errors),
        "n_artists": len(set(s["artist_id"] for s in valid)),
        "n_patterns": len(set(f"{s['artist_id']}_{s['pattern_id']}" for s in valid)),
        "n_gt_files": len([s for s in valid if s["version_id"] == 0]),
        "n_variation_files": len([s for s in valid if s["version_id"] > 0]),
        "n_pairs": sum(p["n_variations"] for p in pattern_stats.values()),
        "duration": {
            "min_s": round(min(durations), 3),
            "max_s": round(max(durations), 3),
            "mean_s": round(float(np.mean(durations)), 3),
            "std_s": round(float(np.std(durations)), 3),
            "median_s": round(float(np.median(durations)), 3),
        },
        "versions_per_pattern": {
            "min": min(vc),
            "max": max(vc),
            "mean": round(float(np.mean(vc)), 2),
            "std": round(float(np.std(vc)), 2),
        },
        "peak_amplitude": {
            "min": round(min(peaks), 4),
            "max": round(max(peaks), 4),
            "mean": round(float(np.mean(peaks)), 4),
            "clipped_files": sum(1 for p in peaks if p >= 0.999),
        },
        "chroma_similarity_gt_vs_variation": {
            "mean": round(float(np.mean(all_sims)), 4),
            "std": round(float(np.std(all_sims)), 4),
            "min": round(float(np.min(all_sims)), 4),
            "max": round(float(np.max(all_sims)), 4),
            "median": round(float(np.median(all_sims)), 4),
            "pct_below_0.7": round(float(np.mean(np.array(all_sims) < 0.7)), 4),
            "pct_above_0.9": round(float(np.mean(np.array(all_sims) > 0.9)), 4),
        },
        "errors": [{"filepath": s["filepath"], "error": s["error"]} for s in errors],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: str, output_dir: str, workers: int = 4) -> None:
    """Run the full dataset analysis pipeline.

    Parameters
    ----------
    data_dir:
        Path to the dopp/ directory containing all WAV files.
    output_dir:
        Directory for all output files.  Created if it does not exist.
    workers:
        Number of parallel worker processes for feature extraction.
        Recommended: 4–8.  Higher values may hit memory limits when loading
        many 7-second audio files simultaneously.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    print("=" * 60)
    print("VarianceEngine — Dataset Analysis — Step 1")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Collect file records
    # ------------------------------------------------------------------
    print("\n[1/5] Collecting file records...")
    records = collect_files(data_dir)
    print(f"      Found {len(records)} WAV files.")

    # ------------------------------------------------------------------
    # 2. Per-file feature extraction (parallel)
    # ------------------------------------------------------------------
    print(f"\n[2/5] Extracting features ({workers} workers)...")
    stats: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, r): r for r in records}
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            stats.append(future.result())

    # Sort for deterministic output
    stats.sort(key=lambda s: (s["artist_id"], s["pattern_id"], s["version_id"]))

    n_errors = sum(1 for s in stats if "error" in s)
    if n_errors:
        print(f"  WARNING: {n_errors} files failed feature extraction.")

    # ------------------------------------------------------------------
    # 3. Build pairs CSV
    # ------------------------------------------------------------------
    print("\n[3/5] Building GT–variation pairs...")
    pairs = build_pairs(records)
    pairs_csv_path = output_dir / "dataset_pairs.csv"
    save_csv(pairs, pairs_csv_path)
    print(f"      {len(pairs)} pairs saved → {pairs_csv_path}")

    # ------------------------------------------------------------------
    # 4. Pattern-level chroma analysis
    # ------------------------------------------------------------------
    print("\n[4/5] Computing pattern-level chroma analysis...")
    pattern_stats = compute_pattern_chroma_stats(stats)

    summary = compute_summary(stats, pattern_stats)

    save_json(stats, output_dir / "dataset_stats.json")
    save_json(pattern_stats, output_dir / "chroma_analysis.json")
    save_json(summary, output_dir / "dataset_summary.json")

    print(f"      {len(pattern_stats)} patterns analysed.")
    print(f"      Outputs saved → {output_dir}/")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\n[5/5] Generating plots...")
    plot_duration_distribution(stats, figures_dir)
    plot_version_count_distribution(stats, figures_dir)
    plot_pitch_range_distribution(stats, figures_dir)
    plot_chroma_similarity_distribution(pattern_stats, figures_dir)

    # ------------------------------------------------------------------
    # Print summary to console
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = summary
    print(f"  Total files      : {s['total_files']} ({s['error_files']} errors)")
    print(f"  Artists          : {s['n_artists']}")
    print(f"  Patterns         : {s['n_patterns']}")
    print(f"  GT–variation pairs: {s['n_pairs']}")
    print(f"  Duration         : {s['duration']['min_s']}–{s['duration']['max_s']}s "
          f"(mean {s['duration']['mean_s']}s ± {s['duration']['std_s']}s)")
    print(f"  Versions/pattern : {s['versions_per_pattern']['min']}–{s['versions_per_pattern']['max']} "
          f"(mean {s['versions_per_pattern']['mean']})")
    print(f"  Clipped files    : {s['peak_amplitude']['clipped_files']}")
    cs = s["chroma_similarity_gt_vs_variation"]
    print(f"  Chroma sim       : mean={cs['mean']}, std={cs['std']}, "
          f"min={cs['min']}, max={cs['max']}")
    print(f"  Structural vars (<0.7 sim): {cs['pct_below_0.7']*100:.1f}%")
    print(f"  Ornamental vars  (>0.9 sim): {cs['pct_above_0.9']*100:.1f}%")
    print("=" * 60)
    print("\nDone. All outputs in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VarianceEngine — Dataset Analysis — Step 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dopp/",
        help="Path to the dopp/ directory containing WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory for all output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for feature extraction.",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.workers)
