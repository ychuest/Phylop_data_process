#!/usr/bin/env python3
"""
Validate chromosome memmap conservation arrays against source bedGraph chunks.

This script checks that encoded per-base scores in binary/<track>/chr*.i16.npy
match the downloaded bedGraph records referenced by coordinate_index.tsv.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OK_STATUSES = {"downloaded", "skipped_existing", "verified_existing"}
_WORKER_ARRAYS: dict[str, np.ndarray] | None = None
_WORKER_META: dict | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate conservation memmap arrays.")
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=SCRIPT_DIR / "hg38_conservation_scores",
    )
    parser.add_argument("--track", default="phyloP100way")
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=None,
        help="Default: <scores-dir>/binary/<track>",
    )
    parser.add_argument(
        "--coordinate-index",
        type=Path,
        default=None,
        help="Default: <scores-dir>/coordinate_index.tsv",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=50,
        help="Number of bedGraph files to validate. Use 0 for all files.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=2000,
        help="Rows sampled per selected bedGraph file. Use 0 for all rows.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of validation worker processes. Default: %(default)s",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help=(
            "Refresh non-interactive progress every N completed files. "
            "Interactive terminals refresh on every file. Default: %(default)s"
        ),
    )
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_meta(binary_dir: Path) -> dict:
    with (binary_dir / "meta.json").open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_index(path: Path, track: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"track", "chrom", "start", "end", "status", "output_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            if row["track"] == track and row["status"] in OK_STATUSES:
                rows.append(row)
    if not rows:
        raise ValueError(f"no usable rows for track={track} in {path}")
    return rows


def encode_score(score: float, scale: float, missing_value: int) -> np.int16:
    info = np.iinfo(np.int16)
    lower = max(info.min, missing_value + 1)
    value = int(round(score * scale))
    value = min(max(value, lower), info.max)
    return np.int16(value)


def load_arrays(binary_dir: Path, meta: dict) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for chrom, rel_path in meta["files"].items():
        path = binary_dir / rel_path
        arr = np.load(path, mmap_mode="r")
        expected = int(meta["chrom_sizes"][chrom])
        if arr.shape != (expected,):
            raise ValueError(f"{path} shape {arr.shape} != {(expected,)}")
        arrays[chrom] = arr
    return arrays


def init_worker(binary_dir: str, meta: dict) -> None:
    global _WORKER_ARRAYS, _WORKER_META
    _WORKER_META = meta
    _WORKER_ARRAYS = load_arrays(Path(binary_dir), meta)


def sample_rows(rows: list[str], count: int, rng: random.Random) -> list[str]:
    if count <= 0 or count >= len(rows):
        return rows
    return rng.sample(rows, count)


def validate_bedgraph_file(
    path: Path,
    arrays: dict[str, np.ndarray],
    scale: float,
    missing_value: int,
    rows_per_file: int,
    rng: random.Random,
) -> tuple[int, int]:
    with open_text(path) as handle:
        data_rows = [line for line in handle if line.strip()]
    selected = sample_rows(data_rows, rows_per_file, rng)
    checked_rows = 0
    checked_bp = 0
    for raw in selected:
        fields = raw.split()
        if len(fields) < 4:
            raise ValueError(f"{path}: malformed bedGraph row: {raw[:100]}")
        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        score = float(fields[3])
        if chrom not in arrays:
            raise KeyError(f"{path}: {chrom} not present in memmap meta")
        expected = encode_score(score, scale=scale, missing_value=missing_value)
        observed = arrays[chrom][start:end]
        if observed.size != end - start:
            raise ValueError(f"{path}: invalid observed slice {chrom}:{start}-{end}")
        if not np.all(observed == expected):
            bad_offset = int(np.flatnonzero(observed != expected)[0])
            bad_coord = start + bad_offset
            raise AssertionError(
                f"mismatch at {chrom}:{bad_coord}; "
                f"observed={int(observed[bad_offset])} expected={int(expected)} "
                f"source={path}"
            )
        checked_rows += 1
        checked_bp += end - start
    return checked_rows, checked_bp


def validate_bedgraph_file_worker(task: tuple[str, int, int]) -> tuple[str, int, int]:
    path_text, rows_per_file, seed = task
    if _WORKER_ARRAYS is None or _WORKER_META is None:
        raise RuntimeError("worker arrays were not initialized")
    path = Path(path_text)
    rng = random.Random(seed)
    try:
        checked_rows, checked_bp = validate_bedgraph_file(
            path=path,
            arrays=_WORKER_ARRAYS,
            scale=float(_WORKER_META["scale"]),
            missing_value=int(_WORKER_META["missing_value"]),
            rows_per_file=rows_per_file,
            rng=rng,
        )
    except Exception as exc:
        raise RuntimeError(f"{path}: {exc}") from exc
    return path_text, checked_rows, checked_bp


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?:??"
    seconds_i = int(round(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


class ProgressReporter:
    def __init__(
        self,
        total_files: int,
        progress_every: int,
        *,
        enabled: bool = True,
        stream=None,
        width: int = 32,
    ) -> None:
        self.total_files = max(0, total_files)
        self.progress_every = max(1, progress_every)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.started = time.monotonic()
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.last_len = 0

    def update(
        self,
        completed_files: int,
        checked_rows: int,
        checked_bp: int,
        *,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        completed_files = min(max(0, completed_files), self.total_files)
        if (
            not force
            and not self.is_tty
            and completed_files % self.progress_every != 0
            and completed_files != self.total_files
        ):
            return

        elapsed = time.monotonic() - self.started
        rate = completed_files / elapsed if elapsed > 0 and completed_files > 0 else 0.0
        remaining = max(0, self.total_files - completed_files)
        eta = remaining / rate if rate > 0 else math.inf
        fraction = completed_files / self.total_files if self.total_files else 1.0
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"[{bar}] {completed_files}/{self.total_files} {fraction * 100:6.2f}% "
            f"| rows={checked_rows} bp={checked_bp} "
            f"| {rate:5.2f} files/s | elapsed {format_duration(elapsed)} "
            f"| ETA {format_duration(eta)}"
        )
        if self.is_tty:
            padding = " " * max(0, self.last_len - len(line))
            self.stream.write("\r" + line + padding)
            self.last_len = len(line)
            if completed_files == self.total_files:
                self.stream.write("\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    binary_dir = args.binary_dir or (args.scores_dir / "binary" / args.track)
    coordinate_index = args.coordinate_index or (args.scores_dir / "coordinate_index.tsv")
    rng = random.Random(args.seed)

    meta = read_meta(binary_dir)
    rows = read_index(coordinate_index, args.track)
    selected_rows = rows if args.files == 0 or args.files >= len(rows) else rng.sample(rows, args.files)
    tasks = [
        (row["output_file"], args.rows_per_file, args.seed + idx)
        for idx, row in enumerate(selected_rows)
    ]

    checked_files = 0
    checked_rows = 0
    checked_bp = 0
    progress = ProgressReporter(
        total_files=len(tasks),
        progress_every=args.progress_every,
        enabled=not args.no_progress_bar,
    )
    progress.update(0, 0, 0, force=True)

    if args.workers == 1:
        init_worker(str(binary_dir), meta)
        for task in tasks:
            _, file_rows, file_bp = validate_bedgraph_file_worker(task)
            checked_files += 1
            checked_rows += file_rows
            checked_bp += file_bp
            progress.update(
                checked_files,
                checked_rows,
                checked_bp,
                force=checked_files == len(tasks),
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(str(binary_dir), meta),
        ) as pool:
            futures = [pool.submit(validate_bedgraph_file_worker, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                _, file_rows, file_bp = future.result()
                checked_files += 1
                checked_rows += file_rows
                checked_bp += file_bp
                progress.update(
                    checked_files,
                    checked_rows,
                    checked_bp,
                    force=checked_files == len(tasks),
                )

    print(f"validated_files={checked_files}")
    print(f"validated_bedgraph_rows={checked_rows}")
    print(f"validated_bp={checked_bp}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
