#!/usr/bin/env python3
"""
Build fast per-chromosome memmap arrays from downloaded hg38 conservation bedGraph.

Input:
  hg38_conservation_scores/coordinate_index.tsv
  hg38_conservation_scores/<track>/bedGraph/*.bedGraph.gz

Output:
  hg38_conservation_scores/binary/<track>/meta.json
  hg38_conservation_scores/binary/<track>/chr1.i16.npy
  ...

The arrays are dense on the chromosome coordinate axis. Missing bases use a
sentinel value, so training can fetch scores by chrom/start/end with O(1)
memmap slicing and no gzip/text parsing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import os
import time
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OK_STATUSES = {"downloaded", "skipped_existing", "verified_existing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert hg38 conservation bedGraph chunks into chromosome memmap arrays."
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=SCRIPT_DIR / "hg38_conservation_scores",
        help="Root output directory from download_hg38_conservation.py.",
    )
    parser.add_argument(
        "--coordinate-index",
        type=Path,
        default=None,
        help="coordinate_index.tsv. Default: <scores-dir>/coordinate_index.tsv",
    )
    parser.add_argument(
        "--fai",
        type=Path,
        default=SCRIPT_DIR / "hg38.ml.fa.fai",
        help="FAI file used for chromosome lengths. Default: %(default)s",
    )
    parser.add_argument("--track", default="phyloP100way")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output binary directory. Default: <scores-dir>/binary/<track>",
    )
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--missing-value", type=int, default=-32768)
    parser.add_argument(
        "--chrom",
        action="append",
        default=None,
        help="Restrict conversion to one chromosome. Can be repeated.",
    )
    parser.add_argument(
        "--max-index-rows",
        type=int,
        default=None,
        help="Process only first N matching coordinate-index rows, useful for smoke tests.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing arrays.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_fai(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least 2 fields")
            sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError(f"no chromosome sizes found in {path}")
    return sizes


def read_coordinate_index(
    path: Path,
    track: str,
    chroms: set[str] | None,
    max_rows: int | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"track", "chrom", "start", "end", "status", "output_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            if row["track"] != track:
                continue
            if chroms is not None and row["chrom"] not in chroms:
                continue
            if row["status"] not in OK_STATUSES:
                continue
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"no usable rows for track={track} in {path}")
    return rows


def rows_by_chrom(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["chrom"], []).append(row)
    for chrom_rows in grouped.values():
        chrom_rows.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return grouped


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def encode_scores(scores: np.ndarray, scale: float, missing_value: int) -> np.ndarray:
    encoded = np.rint(scores * scale).astype(np.int64)
    info = np.iinfo(np.int16)
    lower = max(info.min, missing_value + 1)
    encoded = np.clip(encoded, lower, info.max)
    return encoded.astype(np.int16)


def load_bedgraph_numeric(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open_text(path) as handle:
        data = np.loadtxt(handle, usecols=(1, 2, 3), dtype=np.float64)
    if data.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    if data.ndim == 1:
        data = data.reshape(1, 3)
    starts = data[:, 0].astype(np.int64)
    ends = data[:, 1].astype(np.int64)
    scores = data[:, 2].astype(np.float32)
    return starts, ends, scores


def write_rows_to_array(
    arr: np.memmap,
    rows: list[dict[str, str]],
    scale: float,
    missing_value: int,
) -> tuple[int, int]:
    covered_bp = 0
    file_count = 0
    for row in rows:
        path = Path(row["output_file"])
        if not path.exists():
            raise FileNotFoundError(path)
        starts, ends, scores = load_bedgraph_numeric(path)
        if len(starts) == 0:
            file_count += 1
            continue
        if np.any(starts < 0) or np.any(ends > arr.shape[0]) or np.any(ends <= starts):
            raise ValueError(f"invalid bedGraph coordinates in {path}")

        spans = ends - starts
        encoded = encode_scores(scores, scale=scale, missing_value=missing_value)
        one_base = spans == 1
        if np.any(one_base):
            arr[starts[one_base]] = encoded[one_base]
        for start, end, value in zip(starts[~one_base], ends[~one_base], encoded[~one_base]):
            arr[int(start) : int(end)] = value
        covered_bp += int(spans.sum())
        file_count += 1
    arr.flush()
    return file_count, covered_bp


def write_meta(
    out_dir: Path,
    track: str,
    scale: float,
    missing_value: int,
    chrom_sizes: dict[str, int],
    files: dict[str, str],
    source_index: Path,
) -> None:
    meta = {
        "track": track,
        "dtype": "int16",
        "scale": scale,
        "missing_value": missing_value,
        "chrom_sizes": chrom_sizes,
        "files": files,
        "source_coordinate_index": str(source_index),
        "created_unix_time": int(time.time()),
    }
    tmp = out_dir / "meta.json.tmp"
    with tmp.open("wt", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out_dir / "meta.json")


def main() -> int:
    args = parse_args()
    coordinate_index = args.coordinate_index or (args.scores_dir / "coordinate_index.tsv")
    out_dir = args.out_dir or (args.scores_dir / "binary" / args.track)
    chrom_sizes = read_fai(args.fai)
    requested_chroms = set(args.chrom) if args.chrom else None
    rows = read_coordinate_index(
        coordinate_index,
        track=args.track,
        chroms=requested_chroms,
        max_rows=args.max_index_rows,
    )
    grouped = rows_by_chrom(rows)

    total_bp = sum(int(row["end"]) - int(row["start"]) for row in rows)
    print(f"track={args.track}")
    print(f"coordinate_index={coordinate_index}")
    print(f"output={out_dir}")
    print(f"chromosomes={','.join(grouped)}")
    print(f"index_rows={len(rows)} requested_window_bp={total_bp}")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for chrom, chrom_rows in grouped.items():
        if chrom not in chrom_sizes:
            raise KeyError(f"{chrom} is absent from {args.fai}")
        out_name = f"{chrom}.i16.npy"
        out_path = out_dir / out_name
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} exists; pass --force to overwrite")
        print(f"building {out_path} length={chrom_sizes[chrom]} rows={len(chrom_rows)}")
        arr = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.int16,
            shape=(chrom_sizes[chrom],),
        )
        arr[:] = np.int16(args.missing_value)
        file_count, covered_bp = write_rows_to_array(
            arr,
            chrom_rows,
            scale=args.scale,
            missing_value=args.missing_value,
        )
        print(f"  wrote_files={file_count} covered_bp_with_data={covered_bp}")
        files[chrom] = out_name

    meta_chrom_sizes = {chrom: chrom_sizes[chrom] for chrom in files}
    write_meta(
        out_dir=out_dir,
        track=args.track,
        scale=args.scale,
        missing_value=args.missing_value,
        chrom_sizes=meta_chrom_sizes,
        files=files,
        source_index=coordinate_index,
    )
    print(f"wrote {out_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
