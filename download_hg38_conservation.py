#!/usr/bin/env python3
"""
Download hg38 conservation scores for the unique bases covered by human-sequences.bed.

Default data source:
  UCSC hg38 100-way phyloP bigWig
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw

By default, the script merges overlapping BED intervals, splits the non-redundant
union into fixed-size non-overlapping chunks, extracts those chunks with UCSC
bigWigToBedGraph, and validates that every output bedGraph row stays inside the
requested chunk.

Typical usage:
  cd /data/huda01/yangcheng/projects/work2/data/hg38
  python download_hg38_conservation.py --workers 8

Useful checks:
  python download_hg38_conservation.py --dry-run
  python download_hg38_conservation.py --limit 10 --workers 2
  python download_hg38_conservation.py --verify-only

Important outputs:
  hg38_conservation_scores/merged_intervals.bed
  hg38_conservation_scores/download_windows.bed
  hg38_conservation_scores/coordinate_index.tsv
  hg38_conservation_scores/interval_to_download_windows.tsv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent

TRACK_URLS = {
    "phyloP100way": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
        "phyloP100way/hg38.phyloP100way.bw"
    ),
    "phastCons100way": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
        "phastCons100way/hg38.phastCons100way.bw"
    ),
}

UCSC_BIGWIG_TO_BEDGRAPH_URLS = [
    # Older UCSC binaries are more compatible with this machine's GLIBC 2.27.
    "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v385/bigWigToBedGraph",
    "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v369/bigWigToBedGraph",
    "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigToBedGraph",
]

OK_STATUSES = {"downloaded", "skipped_existing", "verified_existing"}


@dataclass(frozen=True)
class Interval:
    index: int
    bed_line_number: int
    chrom: str
    start: int
    end: int
    name: str
    raw_fields: Tuple[str, ...]

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MergedRegion:
    index: int
    chrom: str
    start: int
    end: int
    source_count: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class BedGraphStats:
    ok: bool
    covered_bp: int = 0
    n_rows: int = 0
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    mean_score: Optional[float] = None
    message: str = ""


@dataclass
class DownloadResult:
    track: str
    interval: Interval
    status: str
    output_file: Path
    covered_bp: Optional[int] = None
    n_rows: Optional[int] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    mean_score: Optional[float] = None
    message: str = ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download UCSC hg38 100-way conservation scores for BED intervals, "
            "one bedGraph file per interval."
        )
    )
    parser.add_argument(
        "--bed",
        type=Path,
        default=SCRIPT_DIR / "human-sequences.bed",
        help="Input BED file. Default: %(default)s",
    )
    parser.add_argument(
        "--fai",
        type=Path,
        default=SCRIPT_DIR / "hg38.ml.fa.fai",
        help="FAI index used to validate BED coordinates. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "hg38_conservation_scores",
        help="Output root directory. Default: %(default)s",
    )
    parser.add_argument(
        "--track",
        choices=("phyloP100way", "phastCons100way", "both"),
        default="phyloP100way",
        help=(
            "Conservation track to extract. phyloP100way is the default "
            "basewise conservation score; phastCons100way is conserved-element "
            "probability."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("unique", "per-bed"),
        default="unique",
        help=(
            "unique merges overlapping BED intervals and downloads each covered "
            "base once; per-bed keeps the old one-output-per-BED-row behavior. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help=(
            "In unique mode, split merged intervals into non-overlapping chunks "
            "of at most this many bp. Use 0 to disable chunking. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent interval extraction tasks. Default: %(default)s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N BED intervals, useful for testing.",
    )
    parser.add_argument(
        "--tool",
        type=Path,
        default=None,
        help="Path to bigWigToBedGraph. If omitted, PATH/local tool is used.",
    )
    parser.add_argument(
        "--tool-url",
        action="append",
        default=None,
        help=(
            "Candidate URL used when auto-downloading bigWigToBedGraph. "
            "Can be repeated. Defaults to UCSC linux.x86_64.v385, v369, then latest."
        ),
    )
    parser.add_argument(
        "--no-tool-download",
        dest="download_tool",
        action="store_false",
        default=True,
        help="Do not auto-download bigWigToBedGraph if it is missing.",
    )
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        default=True,
        help="Write plain .bedGraph files instead of .bedGraph.gz.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download outputs even if existing files pass validation.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient download/extraction failures. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds before one interval extraction command times out. Default: %(default)s",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help=(
            "Refresh non-interactive progress every N completed tasks. "
            "In an interactive terminal the progress bar refreshes on every task. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the progress bar and only print final summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing output files and rewrite the manifest.",
    )
    return parser.parse_args(argv)


def read_fai(path: Path) -> Dict[str, int]:
    chrom_sizes: Dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least 2 fields")
            chrom = fields[0]
            try:
                size = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid chromosome size") from exc
            chrom_sizes[chrom] = size
    if not chrom_sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return chrom_sizes


def read_bed(path: Path, chrom_sizes: Dict[str, int]) -> List[Interval]:
    intervals: List[Interval] = []
    errors: List[str] = []
    with path.open("rt", encoding="utf-8") as handle:
        for bed_line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = tuple(line.rstrip("\n").split("\t"))
            if len(fields) < 3:
                errors.append(f"line {bed_line_number}: expected at least 3 BED fields")
                continue
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                errors.append(f"line {bed_line_number}: start/end are not integers")
                continue
            if chrom not in chrom_sizes:
                errors.append(f"line {bed_line_number}: {chrom} not found in FAI")
                continue
            chrom_size = chrom_sizes[chrom]
            if start < 0:
                errors.append(f"line {bed_line_number}: start < 0")
                continue
            if end <= start:
                errors.append(f"line {bed_line_number}: end <= start")
                continue
            if end > chrom_size:
                errors.append(
                    f"line {bed_line_number}: end {end} exceeds {chrom} size {chrom_size}"
                )
                continue
            name = fields[3] if len(fields) >= 4 and fields[3] else "region"
            intervals.append(
                Interval(
                    index=len(intervals) + 1,
                    bed_line_number=bed_line_number,
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=name,
                    raw_fields=fields,
                )
            )
    if errors:
        preview = "\n".join(errors[:20])
        more = "" if len(errors) <= 20 else f"\n... {len(errors) - 20} more errors"
        raise ValueError(f"Invalid BED coordinates in {path}:\n{preview}{more}")
    if not intervals:
        raise ValueError(f"No intervals found in {path}")
    return intervals


def selected_tracks(track_arg: str) -> List[str]:
    if track_arg == "both":
        return ["phyloP100way", "phastCons100way"]
    return [track_arg]


def chrom_order(chrom_sizes: Dict[str, int]) -> Dict[str, int]:
    return {chrom: rank for rank, chrom in enumerate(chrom_sizes)}


def merge_intervals(
    intervals: Sequence[Interval],
    chrom_sizes: Dict[str, int],
) -> List[MergedRegion]:
    order = chrom_order(chrom_sizes)
    sorted_intervals = sorted(
        intervals,
        key=lambda item: (order.get(item.chrom, 10**9), item.start, item.end),
    )
    merged: List[MergedRegion] = []
    current_chrom: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    source_count = 0

    def flush_current() -> None:
        nonlocal current_chrom, current_start, current_end, source_count
        if current_chrom is None or current_start is None or current_end is None:
            return
        merged.append(
            MergedRegion(
                index=len(merged) + 1,
                chrom=current_chrom,
                start=current_start,
                end=current_end,
                source_count=source_count,
            )
        )

    for interval in sorted_intervals:
        starts_new_region = (
            current_chrom is None
            or interval.chrom != current_chrom
            or interval.start > current_end
        )
        if starts_new_region:
            flush_current()
            current_chrom = interval.chrom
            current_start = interval.start
            current_end = interval.end
            source_count = 1
            continue
        if interval.end > current_end:
            current_end = interval.end
        source_count += 1

    flush_current()
    return merged


def chunk_merged_regions(
    merged_regions: Sequence[MergedRegion],
    chunk_size: int,
) -> List[Interval]:
    if chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")

    download_windows: List[Interval] = []
    for region in merged_regions:
        if chunk_size == 0:
            starts = [region.start]
        else:
            starts = range(region.start, region.end, chunk_size)
        for chunk_number, chunk_start in enumerate(starts, start=1):
            chunk_end = region.end if chunk_size == 0 else min(region.end, chunk_start + chunk_size)
            name = f"unique_region{region.index:06d}_chunk{chunk_number:04d}"
            download_windows.append(
                Interval(
                    index=len(download_windows) + 1,
                    bed_line_number=0,
                    chrom=region.chrom,
                    start=chunk_start,
                    end=chunk_end,
                    name=name,
                    raw_fields=(region.chrom, str(chunk_start), str(chunk_end), name),
                )
            )
    return download_windows


def safe_token(text: str, max_len: int = 80) -> str:
    token = re.sub(r"[^A-Za-z0-9._+-]+", "_", text).strip("._")
    if not token:
        token = "region"
    return token[:max_len]


def output_path(out_dir: Path, track: str, interval: Interval, compress: bool) -> Path:
    suffix = ".bedGraph.gz" if compress else ".bedGraph"
    stem = (
        f"{interval.index:06d}_{interval.chrom}_{interval.start}_"
        f"{interval.end}_{safe_token(interval.name)}_{track}"
    )
    return out_dir / track / "bedGraph" / f"{stem}{suffix}"


def ensure_tool(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.tool is not None:
        tool = args.tool.resolve()
        if not tool.exists():
            raise FileNotFoundError(f"Provided --tool does not exist: {tool}")
        if not os.access(tool, os.X_OK):
            raise PermissionError(f"Provided --tool is not executable: {tool}")
        ok, message = tool_is_runnable(tool)
        if not ok:
            raise RuntimeError(f"Provided --tool is not runnable: {message}")
        return tool

    path_tool = shutil.which("bigWigToBedGraph")
    if path_tool:
        tool = Path(path_tool)
        ok, _ = tool_is_runnable(tool)
        if ok:
            return tool

    local_tool = out_dir / "tools" / "bigWigToBedGraph"
    if local_tool.exists():
        if not os.access(local_tool, os.X_OK):
            local_tool.chmod(local_tool.stat().st_mode | stat.S_IXUSR)
        ok, _ = tool_is_runnable(local_tool)
        if ok:
            return local_tool

    if not args.download_tool:
        raise FileNotFoundError(
            "bigWigToBedGraph was not found. Install it, pass --tool, "
            "or rerun without --no-tool-download."
        )

    local_tool.parent.mkdir(parents=True, exist_ok=True)
    tmp_tool = local_tool.with_suffix(".download")
    candidate_urls = args.tool_url if args.tool_url else UCSC_BIGWIG_TO_BEDGRAPH_URLS
    failed_messages: List[str] = []
    for tool_url in candidate_urls:
        if tmp_tool.exists():
            tmp_tool.unlink()
        try:
            download_file(tool_url, tmp_tool)
            mode = tmp_tool.stat().st_mode
            tmp_tool.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            ok, message = tool_is_runnable(tmp_tool)
            if ok:
                os.replace(tmp_tool, local_tool)
                return local_tool
            failed_messages.append(f"{tool_url}: {message}")
        except Exception as exc:  # noqa: BLE001 - report all candidate failures.
            failed_messages.append(f"{tool_url}: {exc}")
    if tmp_tool.exists():
        tmp_tool.unlink()
    raise RuntimeError(
        "Could not download a runnable bigWigToBedGraph. Tried:\n"
        + "\n".join(failed_messages)
    )


def tool_is_runnable(tool: Path) -> Tuple[bool, str]:
    try:
        completed = subprocess.run(
            [str(tool)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001 - this is diagnostic.
        return False, str(exc)
    text = (completed.stdout or "") + (completed.stderr or "")
    if "GLIBC_" in text or "not found" in text or "Permission denied" in text:
        return False, text.strip()[:1000]
    if "bigWigToBedGraph" in text and "usage:" in text:
        return True, "usage check passed"
    return False, text.strip()[:1000] or f"unexpected exit {completed.returncode}"


def download_file(url: str, output: Path, timeout: int = 60) -> None:
    request = Request(url, headers={"User-Agent": "hg38-conservation-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response, output.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if output.stat().st_size < 100_000:
        raise RuntimeError(f"Downloaded tool from {url} is unexpectedly small")


def open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def verify_bedgraph(path: Path, interval: Interval) -> BedGraphStats:
    covered_bp = 0
    n_rows = 0
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    weighted_sum = 0.0
    previous_end: Optional[int] = None

    try:
        with open_text_maybe_gzip(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("track") or stripped.startswith("browser"):
                    return BedGraphStats(
                        ok=False,
                        message=f"line {line_number}: unexpected track/browser line",
                    )
                fields = stripped.split()
                if len(fields) < 4:
                    return BedGraphStats(
                        ok=False,
                        message=f"line {line_number}: expected 4 bedGraph fields",
                    )
                chrom = fields[0]
                try:
                    row_start = int(fields[1])
                    row_end = int(fields[2])
                    score = float(fields[3])
                except ValueError:
                    return BedGraphStats(
                        ok=False,
                        message=f"line {line_number}: invalid start/end/score",
                    )
                if not math.isfinite(score):
                    return BedGraphStats(
                        ok=False,
                        message=f"line {line_number}: score is not finite",
                    )
                if chrom != interval.chrom:
                    return BedGraphStats(
                        ok=False,
                        message=(
                            f"line {line_number}: chrom {chrom} != {interval.chrom}"
                        ),
                    )
                if row_start < interval.start or row_end > interval.end:
                    return BedGraphStats(
                        ok=False,
                        message=(
                            f"line {line_number}: row {chrom}:{row_start}-{row_end} "
                            f"outside requested {interval.chrom}:"
                            f"{interval.start}-{interval.end}"
                        ),
                    )
                if row_start >= row_end:
                    return BedGraphStats(
                        ok=False,
                        message=f"line {line_number}: row_start >= row_end",
                    )
                if previous_end is not None and row_start < previous_end:
                    return BedGraphStats(
                        ok=False,
                        message=f"line {line_number}: overlapping or unsorted rows",
                    )
                span = row_end - row_start
                n_rows += 1
                covered_bp += span
                weighted_sum += score * span
                min_score = score if min_score is None else min(min_score, score)
                max_score = score if max_score is None else max(max_score, score)
                previous_end = row_end
    except OSError as exc:
        return BedGraphStats(ok=False, message=f"could not read output: {exc}")

    if covered_bp > interval.length:
        return BedGraphStats(
            ok=False,
            covered_bp=covered_bp,
            n_rows=n_rows,
            min_score=min_score,
            max_score=max_score,
            message=(
                f"covered_bp {covered_bp} exceeds requested length {interval.length}"
            ),
        )

    mean_score = weighted_sum / covered_bp if covered_bp else None
    return BedGraphStats(
        ok=True,
        covered_bp=covered_bp,
        n_rows=n_rows,
        min_score=min_score,
        max_score=max_score,
        mean_score=mean_score,
    )


def compress_to_gzip(raw_path: Path, gzip_path: Path) -> None:
    tmp_gzip = gzip_path.with_suffix(gzip_path.suffix + ".tmp")
    with raw_path.open("rb") as src, gzip.open(tmp_gzip, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(tmp_gzip, gzip_path)
    raw_path.unlink()


def run_one_interval(
    interval: Interval,
    track: str,
    bw_url: str,
    tool: Path,
    out_dir: Path,
    compress: bool,
    force: bool,
    retries: int,
    timeout: int,
) -> DownloadResult:
    final_path = output_path(out_dir, track, interval, compress)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not force:
        stats = verify_bedgraph(final_path, interval)
        if stats.ok:
            return DownloadResult(
                track=track,
                interval=interval,
                status="skipped_existing",
                output_file=final_path,
                covered_bp=stats.covered_bp,
                n_rows=stats.n_rows,
                min_score=stats.min_score,
                max_score=stats.max_score,
                mean_score=stats.mean_score,
                message="existing output passed coordinate validation",
            )

    tmp_dir = out_dir / track / "tmp"
    logs_dir = out_dir / track / "logs"
    cache_dir = out_dir / "ucsc_udc_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_raw = tmp_dir / f"{final_path.name}.tmp.{os.getpid()}.bedGraph"
    log_path = logs_dir / f"{final_path.name}.stderr.txt"
    command = [
        str(tool),
        bw_url,
        str(tmp_raw),
        f"-chrom={interval.chrom}",
        f"-start={interval.start}",
        f"-end={interval.end}",
        f"-udcDir={cache_dir}",
    ]

    last_message = ""
    for attempt in range(1, retries + 2):
        if tmp_raw.exists():
            tmp_raw.unlink()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_message = f"timeout after {timeout}s on attempt {attempt}"
            time.sleep(min(30, 2 * attempt))
            continue
        except OSError as exc:
            last_message = f"could not run bigWigToBedGraph: {exc}"
            break

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            last_message = (
                f"bigWigToBedGraph exit {completed.returncode} on attempt "
                f"{attempt}; stderr={stderr[:500]} stdout={stdout[:500]}"
            )
            time.sleep(min(30, 2 * attempt))
            continue

        if not tmp_raw.exists():
            last_message = f"bigWigToBedGraph produced no output file on attempt {attempt}"
            time.sleep(min(30, 2 * attempt))
            continue

        stats = verify_bedgraph(tmp_raw, interval)
        if not stats.ok:
            last_message = f"coordinate validation failed: {stats.message}"
            time.sleep(min(30, 2 * attempt))
            continue

        if compress:
            compress_to_gzip(tmp_raw, final_path)
        else:
            os.replace(tmp_raw, final_path)

        if log_path.exists():
            log_path.unlink()
        return DownloadResult(
            track=track,
            interval=interval,
            status="downloaded",
            output_file=final_path,
            covered_bp=stats.covered_bp,
            n_rows=stats.n_rows,
            min_score=stats.min_score,
            max_score=stats.max_score,
            mean_score=stats.mean_score,
            message="downloaded and coordinate-validated",
        )

    log_path.write_text(last_message + "\n", encoding="utf-8")
    return DownloadResult(
        track=track,
        interval=interval,
        status="failed",
        output_file=final_path,
        message=last_message,
    )


def verify_existing_interval(
    interval: Interval,
    track: str,
    out_dir: Path,
    compress: bool,
) -> DownloadResult:
    final_path = output_path(out_dir, track, interval, compress)
    if not final_path.exists():
        return DownloadResult(
            track=track,
            interval=interval,
            status="missing",
            output_file=final_path,
            message="expected output file is missing",
        )
    stats = verify_bedgraph(final_path, interval)
    if not stats.ok:
        return DownloadResult(
            track=track,
            interval=interval,
            status="failed",
            output_file=final_path,
            message=f"coordinate validation failed: {stats.message}",
        )
    return DownloadResult(
        track=track,
        interval=interval,
        status="verified_existing",
        output_file=final_path,
        covered_bp=stats.covered_bp,
        n_rows=stats.n_rows,
        min_score=stats.min_score,
        max_score=stats.max_score,
        mean_score=stats.mean_score,
        message="existing output passed coordinate validation",
    )


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.8g}"


def format_optional_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(value)


def clean_message(message: str) -> str:
    return re.sub(r"\s+", " ", message).strip()


def write_manifest(path: Path, results: Iterable[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    header = [
        "track",
        "interval_index",
        "bed_line_number",
        "chrom",
        "start",
        "end",
        "bed_name",
        "expected_bp",
        "status",
        "covered_bp",
        "coverage_fraction",
        "n_rows",
        "min_score",
        "max_score",
        "mean_score",
        "output_file",
        "message",
    ]
    sorted_results = sorted(
        results, key=lambda item: (item.track, item.interval.index)
    )
    with tmp_path.open("wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for result in sorted_results:
            interval = result.interval
            coverage_fraction = (
                "NA"
                if result.covered_bp is None
                else f"{result.covered_bp / interval.length:.8f}"
            )
            row = [
                result.track,
                str(interval.index),
                str(interval.bed_line_number),
                interval.chrom,
                str(interval.start),
                str(interval.end),
                interval.name,
                str(interval.length),
                result.status,
                format_optional_int(result.covered_bp),
                coverage_fraction,
                format_optional_int(result.n_rows),
                format_optional_float(result.min_score),
                format_optional_float(result.max_score),
                format_optional_float(result.mean_score),
                str(result.output_file),
                clean_message(result.message),
            ]
            handle.write("\t".join(row) + "\n")
    os.replace(tmp_path, path)


def write_merged_regions(path: Path, merged_regions: Sequence[MergedRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", encoding="utf-8") as handle:
        for region in merged_regions:
            handle.write(
                "\t".join(
                    [
                        region.chrom,
                        str(region.start),
                        str(region.end),
                        f"merged_region_{region.index:06d}",
                        str(region.source_count),
                    ]
                )
                + "\n"
            )
    os.replace(tmp_path, path)


def write_download_windows(path: Path, intervals: Sequence[Interval]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", encoding="utf-8") as handle:
        for interval in intervals:
            handle.write(
                "\t".join(
                    [
                        interval.chrom,
                        str(interval.start),
                        str(interval.end),
                        interval.name,
                        str(interval.index),
                    ]
                )
                + "\n"
            )
    os.replace(tmp_path, path)


def write_interval_to_windows(
    path: Path,
    original_intervals: Sequence[Interval],
    download_windows: Sequence[Interval],
    chrom_sizes: Dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = chrom_order(chrom_sizes)
    originals = sorted(
        original_intervals,
        key=lambda item: (order.get(item.chrom, 10**9), item.start, item.end),
    )
    windows_by_chrom: Dict[str, List[Interval]] = {}
    for window in download_windows:
        windows_by_chrom.setdefault(window.chrom, []).append(window)
    for chrom_windows in windows_by_chrom.values():
        chrom_windows.sort(key=lambda item: (item.start, item.end))

    pointers = {chrom: 0 for chrom in windows_by_chrom}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    header = [
        "original_interval_index",
        "bed_line_number",
        "chrom",
        "start",
        "end",
        "bed_name",
        "download_window_index",
        "window_start",
        "window_end",
        "overlap_start",
        "overlap_end",
        "overlap_bp",
    ]
    with tmp_path.open("wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for interval in originals:
            chrom_windows = windows_by_chrom.get(interval.chrom, [])
            pointer = pointers.get(interval.chrom, 0)
            while pointer < len(chrom_windows) and chrom_windows[pointer].end <= interval.start:
                pointer += 1
            pointers[interval.chrom] = pointer

            window_index = pointer
            covered_bp = 0
            while (
                window_index < len(chrom_windows)
                and chrom_windows[window_index].start < interval.end
            ):
                window = chrom_windows[window_index]
                overlap_start = max(interval.start, window.start)
                overlap_end = min(interval.end, window.end)
                if overlap_start < overlap_end:
                    overlap_bp = overlap_end - overlap_start
                    covered_bp += overlap_bp
                    row = [
                        str(interval.index),
                        str(interval.bed_line_number),
                        interval.chrom,
                        str(interval.start),
                        str(interval.end),
                        interval.name,
                        str(window.index),
                        str(window.start),
                        str(window.end),
                        str(overlap_start),
                        str(overlap_end),
                        str(overlap_bp),
                    ]
                    handle.write("\t".join(row) + "\n")
                window_index += 1
            if covered_bp != interval.length:
                raise RuntimeError(
                    f"Download windows cover {covered_bp} bp, expected "
                    f"{interval.length} bp for original interval "
                    f"{interval.chrom}:{interval.start}-{interval.end}"
                )
    os.replace(tmp_path, path)


def write_coordinate_index(path: Path, results: Iterable[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    header = [
        "track",
        "download_window_index",
        "chrom",
        "start",
        "end",
        "expected_bp",
        "status",
        "covered_bp",
        "coverage_fraction",
        "output_file",
    ]
    sorted_results = sorted(
        results, key=lambda item: (item.track, item.interval.index)
    )
    with tmp_path.open("wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for result in sorted_results:
            interval = result.interval
            coverage_fraction = (
                "NA"
                if result.covered_bp is None
                else f"{result.covered_bp / interval.length:.8f}"
            )
            row = [
                result.track,
                str(interval.index),
                interval.chrom,
                str(interval.start),
                str(interval.end),
                str(interval.length),
                result.status,
                format_optional_int(result.covered_bp),
                coverage_fraction,
                str(result.output_file),
            ]
            handle.write("\t".join(row) + "\n")
    os.replace(tmp_path, path)


def print_plan(
    original_intervals: Sequence[Interval],
    merged_regions: Sequence[MergedRegion],
    download_windows: Sequence[Interval],
    tracks: Sequence[str],
    out_dir: Path,
    compress: bool,
    mode: str,
) -> None:
    original_bp = sum(interval.length for interval in original_intervals)
    download_bp = sum(interval.length for interval in download_windows)
    print(f"Mode: {mode}")
    print(f"Input BED intervals: {len(original_intervals)}")
    print(f"Input BED bp with overlaps counted: {original_bp}")
    if merged_regions:
        print(f"Merged non-overlapping regions: {len(merged_regions)}")
        print(f"Unique bp to download: {download_bp}")
    print(f"Download windows: {len(download_windows)}")
    print(f"Total requested bp for download: {download_bp}")
    print(f"Tracks: {', '.join(tracks)}")
    print(f"Output root: {out_dir}")
    print(f"Compression: {'gzip' if compress else 'plain bedGraph'}")
    first = download_windows[0]
    for track in tracks:
        print(f"Example output for {track}: {output_path(out_dir, track, first, compress)}")


def summarize_results(results: Sequence[DownloadResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


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
        total: int,
        progress_every: int,
        *,
        enabled: bool = True,
        stream=None,
        width: int = 32,
    ) -> None:
        self.total = max(0, total)
        self.progress_every = max(1, progress_every)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.started = time.monotonic()
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.last_len = 0

    def update(self, completed: int, counts: Dict[str, int], *, force: bool = False) -> None:
        if not self.enabled:
            return
        completed = min(max(0, completed), self.total)
        if not force and not self.is_tty and completed % self.progress_every != 0 and completed != self.total:
            return

        elapsed = time.monotonic() - self.started
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        remaining = max(0, self.total - completed)
        eta = remaining / rate if rate > 0 else math.inf
        fraction = completed / self.total if self.total else 1.0
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        count_text = " ".join(
            f"{status}={counts.get(status, 0)}"
            for status in (
                "downloaded",
                "skipped_existing",
                "verified_existing",
                "failed",
                "missing",
            )
            if counts.get(status, 0)
        )
        if not count_text:
            count_text = "waiting=0"
        line = (
            f"[{bar}] {completed}/{self.total} {fraction * 100:6.2f}% "
            f"| {count_text} | {rate:5.2f} tasks/s "
            f"| elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
        )
        if self.is_tty:
            padding = " " * max(0, self.last_len - len(line))
            self.stream.write("\r" + line + padding)
            self.last_len = len(line)
            if completed == self.total:
                self.stream.write("\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")

    chrom_sizes = read_fai(args.fai)
    original_intervals = read_bed(args.bed, chrom_sizes)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        original_intervals = original_intervals[: args.limit]
    tracks = selected_tracks(args.track)

    if args.mode == "unique":
        merged_regions = merge_intervals(original_intervals, chrom_sizes)
        download_windows = chunk_merged_regions(merged_regions, args.chunk_size)
    else:
        merged_regions = []
        download_windows = original_intervals

    print_plan(
        original_intervals=original_intervals,
        merged_regions=merged_regions,
        download_windows=download_windows,
        tracks=tracks,
        out_dir=args.out_dir,
        compress=args.compress,
        mode=args.mode,
    )
    sys.stdout.flush()
    if args.dry_run:
        print("Dry run complete. No downloads were started.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_download_windows(args.out_dir / "download_windows.bed", download_windows)
    if args.mode == "unique":
        write_merged_regions(args.out_dir / "merged_intervals.bed", merged_regions)
        write_interval_to_windows(
            path=args.out_dir / "interval_to_download_windows.tsv",
            original_intervals=original_intervals,
            download_windows=download_windows,
            chrom_sizes=chrom_sizes,
        )

    results: List[DownloadResult] = []
    total_tasks = len(download_windows) * len(tracks)
    progress_counts: Dict[str, int] = {}
    progress = ProgressReporter(
        total_tasks,
        args.progress_every,
        enabled=not args.no_progress_bar,
    )
    progress.update(0, {}, force=True)

    if args.verify_only:
        completed_count = 0
        for track in tracks:
            for interval in download_windows:
                results.append(
                    verify_existing_interval(
                        interval=interval,
                        track=track,
                        out_dir=args.out_dir,
                        compress=args.compress,
                    )
                )
                progress_counts[results[-1].status] = progress_counts.get(results[-1].status, 0) + 1
                completed_count += 1
                progress.update(
                    completed_count,
                    progress_counts,
                    force=completed_count == total_tasks,
                )
    else:
        tool = ensure_tool(args, args.out_dir)
        print(f"Using bigWigToBedGraph: {tool}", flush=True)

        futures = []
        completed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for track in tracks:
                bw_url = TRACK_URLS[track]
                for interval in download_windows:
                    futures.append(
                        pool.submit(
                            run_one_interval,
                            interval,
                            track,
                            bw_url,
                            tool,
                            args.out_dir,
                            args.compress,
                            args.force,
                            args.retries,
                            args.timeout,
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                progress_counts[result.status] = progress_counts.get(result.status, 0) + 1
                completed_count += 1
                progress.update(
                    completed_count,
                    progress_counts,
                    force=completed_count == total_tasks or result.status == "failed",
                )

    manifest_path = args.out_dir / "manifest.tsv"
    write_manifest(manifest_path, results)
    coordinate_index_path = args.out_dir / "coordinate_index.tsv"
    write_coordinate_index(coordinate_index_path, results)
    counts = summarize_results(results)
    print(f"Manifest written: {manifest_path}")
    print(f"Coordinate index written: {coordinate_index_path}")
    print(
        "Final status counts: "
        + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    )

    failures = [
        result
        for result in results
        if result.status not in OK_STATUSES
    ]
    if failures:
        print(
            f"Found {len(failures)} missing/failed intervals. "
            f"Inspect {manifest_path} and per-track logs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
