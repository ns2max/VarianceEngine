"""
io.py — Filename parsing and structured output utilities.

Project: VarianceEngine

The DOPP dataset uses the naming convention:
    {artistID}_{patternID}_{versionID}.wav

All identifiers are integers.  versionID=0 is the ground truth; all other
versions are expressive variations.

Design decisions
----------------
- Parsing is done with a single regex rather than a split on '_' because
  artistID, patternID, and versionID are all numeric and an underscore-based
  split would misparse if any ID ever contained an underscore (defensive).
- Path objects are returned as strings in CSVs for portability across OS.
"""

import re
import json
import csv
from pathlib import Path


# Matches: {int}_{int}_{int}.wav — nothing more, nothing less.
_FILENAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.wav$")


def parse_filename(filename: str) -> dict | None:
    """Parse a DOPP dataset filename into its constituent IDs.

    Parameters
    ----------
    filename:
        Bare filename (not a full path), e.g. '40_3_7.wav'.

    Returns
    -------
    dict with keys artist_id, pattern_id, version_id (all int), or None if
    the filename does not match the expected pattern.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    return {
        "artist_id": int(m.group(1)),
        "pattern_id": int(m.group(2)),
        "version_id": int(m.group(3)),
    }


def collect_files(data_dir: str | Path) -> list[dict]:
    """Walk data_dir and return a sorted list of parsed file records.

    Each record is a dict:
        artist_id   : int
        pattern_id  : int
        version_id  : int
        filepath    : str   — absolute path

    Unparseable filenames are silently skipped (logged to stderr if any found).

    Returns records sorted by (artist_id, pattern_id, version_id) for
    deterministic processing order.
    """
    data_dir = Path(data_dir)
    records = []
    skipped = []

    for wav_path in data_dir.glob("*.wav"):
        parsed = parse_filename(wav_path.name)
        if parsed is None:
            skipped.append(wav_path.name)
            continue
        parsed["filepath"] = str(wav_path.resolve())
        records.append(parsed)

    if skipped:
        import sys
        print(f"[io] WARNING: skipped {len(skipped)} unparseable files: {skipped[:5]}...", file=sys.stderr)

    records.sort(key=lambda r: (r["artist_id"], r["pattern_id"], r["version_id"]))
    return records


def build_pairs(records: list[dict]) -> list[dict]:
    """From a flat list of file records, build (ground_truth, variation) pairs.

    A pair is formed for every record where version_id > 0, paired with the
    version_id=0 record from the same (artist_id, pattern_id) group.

    Patterns that have no version_id=0 file are excluded with a warning.

    Returns
    -------
    List of dicts, each with keys:
        artist_id, pattern_id,
        gt_filepath, gt_version_id (always 0),
        var_filepath, var_version_id
    """
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (r["artist_id"], r["pattern_id"])
        groups[key].append(r)

    pairs = []
    missing_gt = []

    for (artist_id, pattern_id), group in sorted(groups.items()):
        gt_records = [r for r in group if r["version_id"] == 0]
        var_records = [r for r in group if r["version_id"] > 0]

        if not gt_records:
            missing_gt.append((artist_id, pattern_id))
            continue

        gt = gt_records[0]
        for var in var_records:
            pairs.append({
                "artist_id": artist_id,
                "pattern_id": pattern_id,
                "gt_filepath": gt["filepath"],
                "gt_version_id": 0,
                "var_filepath": var["filepath"],
                "var_version_id": var["version_id"],
            })

    if missing_gt:
        import sys
        print(f"[io] WARNING: {len(missing_gt)} patterns have no version_id=0: {missing_gt[:5]}", file=sys.stderr)

    return pairs


def save_csv(records: list[dict], output_path: str | Path) -> None:
    """Write a list of dicts to a CSV file.

    Column order follows the key order of the first record.  Output is UTF-8
    with Unix line endings for cross-platform compatibility.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("Cannot write empty records list to CSV.")

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_json(data: dict | list, output_path: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file with consistent formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, allow_nan=False)
