#!/usr/bin/env python3
"""
Summarize and plot hg38 phyloP100way score distributions per chromosome.

Input:
  hg38_conservation_scores/binary/phyloP100way/meta.json
  hg38_conservation_scores/binary/phyloP100way/chr*.i16.npy

Output:
  phyloP100way_distribution_plots/*.png
  phyloP100way_distribution_plots/*.pdf
  phyloP100way_distribution_plots/phyloP100way_chromosome_stats.tsv
  phyloP100way_distribution_report.md

The .npy arrays store encoded scores:
  real_score = encoded_value / scale
  missing_value is excluded from all distribution statistics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BINARY_DIR = SCRIPT_DIR / "hg38_conservation_scores" / "binary" / "phyloP100way"
DEFAULT_PLOT_DIR = SCRIPT_DIR / "phyloP100way_distribution_plots"
DEFAULT_REPORT = SCRIPT_DIR / "phyloP100way_distribution_report.md"
INT16_OFFSET = 32768
INT16_BINS = 65536


@dataclass
class ChromStats:
    chrom: str
    chrom_size: int
    valid_bases: int
    missing_bases: int
    coverage_fraction: float
    min_score: float
    q001: float
    q01: float
    q05: float
    q25: float
    median: float
    q75: float
    q95: float
    q99: float
    q999: float
    max_score: float
    mean: float
    std: float
    frac_negative: float
    frac_zero: float
    frac_positive: float
    frac_ge_1: float
    frac_ge_2: float
    frac_ge_3: float
    frac_le_minus1: float
    frac_le_minus2: float
    frac_le_minus3: float
    png_path: str
    pdf_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-chromosome hg38 phyloP100way distribution statistics."
    )
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=DEFAULT_BINARY_DIR,
        help="Directory with meta.json and chr*.i16.npy. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory for PNG/PDF plots and TSV stats. Default: %(default)s",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown report path. Default: %(default)s",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=23,
        help="Number of chromosome worker processes. Default: %(default)s",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000_000,
        help="Number of encoded bases read per chunk. Default: %(default)s",
    )
    parser.add_argument(
        "--chrom",
        action="append",
        default=None,
        help="Restrict to one chromosome. Can be repeated. Default: all chromosomes in meta.json.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=240,
        help="Histogram bins for plotting central distribution. Default: %(default)s",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Refresh non-interactive progress every N completed chromosomes. Default: %(default)s",
    )
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def natural_chrom_key(chrom: str) -> tuple[int, int | str]:
    token = chrom.removeprefix("chr")
    if token.isdigit():
        return (0, int(token))
    if token == "X":
        return (1, 23)
    if token == "Y":
        return (1, 24)
    if token in {"M", "MT"}:
        return (1, 25)
    return (2, token)


def read_meta(binary_dir: Path) -> dict:
    with (binary_dir / "meta.json").open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def encoded_counts(path: Path, missing_value: int, chunk_size: int) -> tuple[np.ndarray, int, int]:
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != np.int16:
        raise ValueError(f"{path} dtype {arr.dtype}; expected int16")
    counts = np.zeros(INT16_BINS, dtype=np.int64)
    valid_bases = 0
    missing_bases = 0
    for start in range(0, arr.shape[0], chunk_size):
        chunk = np.asarray(arr[start : start + chunk_size])
        valid = chunk != missing_value
        missing_bases += int((~valid).sum())
        if valid.any():
            values = chunk[valid].astype(np.int32) + INT16_OFFSET
            counts += np.bincount(values, minlength=INT16_BINS)
            valid_bases += int(valid.sum())
    return counts, valid_bases, missing_bases


def quantile_from_counts(counts: np.ndarray, q: float, scale: float) -> float:
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    q = min(max(q, 0.0), 1.0)
    rank = int(math.ceil(q * total))
    rank = max(rank, 1)
    idx = int(np.searchsorted(np.cumsum(counts), rank, side="left"))
    encoded = idx - INT16_OFFSET
    return encoded / scale


def score_fraction(counts: np.ndarray, predicate: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    return float(counts[predicate].sum() / total)


def stats_from_counts(
    chrom: str,
    chrom_size: int,
    counts: np.ndarray,
    valid_bases: int,
    missing_bases: int,
    scale: float,
    png_path: Path,
    pdf_path: Path,
) -> ChromStats:
    if valid_bases == 0:
        raise ValueError(f"{chrom}: no non-missing phyloP scores")
    encoded = np.arange(INT16_BINS, dtype=np.int32) - INT16_OFFSET
    present = counts > 0
    present_encoded = encoded[present]
    score_values = present_encoded.astype(np.float64) / scale
    present_counts = counts[present].astype(np.float64)
    encoded_mean = float((present_encoded.astype(np.float64) * present_counts).sum() / valid_bases)
    encoded_second = float(((present_encoded.astype(np.float64) ** 2) * present_counts).sum() / valid_bases)
    encoded_var = max(0.0, encoded_second - encoded_mean**2)
    score_axis = encoded.astype(np.float64) / scale

    return ChromStats(
        chrom=chrom,
        chrom_size=chrom_size,
        valid_bases=valid_bases,
        missing_bases=missing_bases,
        coverage_fraction=valid_bases / chrom_size if chrom_size else float("nan"),
        min_score=float(score_values[0]),
        q001=quantile_from_counts(counts, 0.001, scale),
        q01=quantile_from_counts(counts, 0.01, scale),
        q05=quantile_from_counts(counts, 0.05, scale),
        q25=quantile_from_counts(counts, 0.25, scale),
        median=quantile_from_counts(counts, 0.50, scale),
        q75=quantile_from_counts(counts, 0.75, scale),
        q95=quantile_from_counts(counts, 0.95, scale),
        q99=quantile_from_counts(counts, 0.99, scale),
        q999=quantile_from_counts(counts, 0.999, scale),
        max_score=float(score_values[-1]),
        mean=encoded_mean / scale,
        std=math.sqrt(encoded_var) / scale,
        frac_negative=score_fraction(counts, score_axis < 0),
        frac_zero=score_fraction(counts, score_axis == 0),
        frac_positive=score_fraction(counts, score_axis > 0),
        frac_ge_1=score_fraction(counts, score_axis >= 1),
        frac_ge_2=score_fraction(counts, score_axis >= 2),
        frac_ge_3=score_fraction(counts, score_axis >= 3),
        frac_le_minus1=score_fraction(counts, score_axis <= -1),
        frac_le_minus2=score_fraction(counts, score_axis <= -2),
        frac_le_minus3=score_fraction(counts, score_axis <= -3),
        png_path=str(png_path),
        pdf_path=str(pdf_path),
    )


def configure_matplotlib() -> None:
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-hg38-phyloP"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def plot_distribution(
    chrom: str,
    counts: np.ndarray,
    stats: ChromStats,
    scale: float,
    bins: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoded = np.arange(INT16_BINS, dtype=np.int32) - INT16_OFFSET
    score_axis = encoded.astype(np.float64) / scale
    present = counts > 0
    values = score_axis[present]
    weights = counts[present].astype(np.float64)

    central_min = stats.q001
    central_max = stats.q999
    if not math.isfinite(central_min) or not math.isfinite(central_max) or central_min >= central_max:
        central_min, central_max = stats.min_score, stats.max_score
    pad = max((central_max - central_min) * 0.05, 0.1)
    x_min = central_min - pad
    x_max = central_max + pad
    central_mask = (values >= x_min) & (values <= x_max)

    hist_counts, edges = np.histogram(values[central_mask], bins=bins, range=(x_min, x_max), weights=weights[central_mask])
    bin_widths = np.diff(edges)
    density = hist_counts / max(hist_counts.sum(), 1.0) / bin_widths
    centers = 0.5 * (edges[:-1] + edges[1:])

    cumulative = np.cumsum(weights) / max(weights.sum(), 1.0)
    stats_text = "\n".join(
        [
            f"valid bases: {stats.valid_bases:,}",
            f"missing bases: {stats.missing_bases:,}",
            f"coverage: {stats.coverage_fraction:.3%}",
            f"mean: {stats.mean:.3f}",
            f"median: {stats.median:.3f}",
            f"IQR: {stats.q25:.3f} to {stats.q75:.3f}",
            f"P1/P99: {stats.q01:.3f} / {stats.q99:.3f}",
            f">0: {stats.frac_positive:.2%}",
            f"<0: {stats.frac_negative:.2%}",
        ]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.2),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    ax = axes[0]
    ax.bar(centers, density, width=bin_widths, color="#4C78A8", alpha=0.82, linewidth=0)
    ax.axvline(0, color="#222222", linewidth=1.0, linestyle=":", label="neutral score 0")
    ax.axvline(stats.median, color="#D62728", linewidth=1.8, label=f"median={stats.median:.3f}")
    ax.axvline(stats.mean, color="#2CA02C", linewidth=1.5, linestyle="--", label=f"mean={stats.mean:.3f}")
    ax.axvspan(stats.q25, stats.q75, color="#D62728", alpha=0.12, label="IQR")
    ax.set_title(f"{chrom} phyloP100way distribution", fontsize=14)
    ax.set_xlabel("phyloP100way score")
    ax.set_ylabel("density")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    ax_cdf = axes[1]
    ax_cdf.plot(values, cumulative, color="#4C78A8", linewidth=1.6)
    ax_cdf.axvline(stats.median, color="#D62728", linewidth=1.4)
    ax_cdf.axvline(0, color="#222222", linewidth=1.0, linestyle=":")
    ax_cdf.set_xlim(x_min, x_max)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel("phyloP100way score")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.grid(alpha=0.25)

    fig.suptitle("Positive values indicate conservation; negative values indicate acceleration.", fontsize=10)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def analyze_chromosome(task: dict) -> ChromStats:
    chrom = task["chrom"]
    binary_dir = Path(task["binary_dir"])
    rel_path = task["rel_path"]
    chrom_size = int(task["chrom_size"])
    scale = float(task["scale"])
    missing_value = int(task["missing_value"])
    chunk_size = int(task["chunk_size"])
    bins = int(task["bins"])
    out_dir = Path(task["out_dir"])

    png_path = out_dir / f"{chrom}.phyloP100way.distribution.png"
    pdf_path = out_dir / f"{chrom}.phyloP100way.distribution.pdf"
    counts, valid_bases, missing_bases = encoded_counts(binary_dir / rel_path, missing_value, chunk_size)
    stats = stats_from_counts(
        chrom=chrom,
        chrom_size=chrom_size,
        counts=counts,
        valid_bases=valid_bases,
        missing_bases=missing_bases,
        scale=scale,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    plot_distribution(
        chrom=chrom,
        counts=counts,
        stats=stats,
        scale=scale,
        bins=bins,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    return stats


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
    def __init__(self, total: int, progress_every: int, *, enabled: bool = True) -> None:
        self.total = total
        self.progress_every = max(1, progress_every)
        self.enabled = enabled
        self.started = time.monotonic()
        self.is_tty = sys.stderr.isatty()
        self.last_len = 0

    def update(self, completed: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and not self.is_tty and completed % self.progress_every != 0 and completed != self.total:
            return
        elapsed = time.monotonic() - self.started
        rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
        eta = (self.total - completed) / rate if rate > 0 else math.inf
        frac = completed / self.total if self.total else 1.0
        width = 30
        filled = int(round(width * frac))
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"[{bar}] {completed}/{self.total} {frac * 100:6.2f}% "
            f"| {rate:.2f} chrom/s | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
        )
        if self.is_tty:
            padding = " " * max(0, self.last_len - len(line))
            sys.stderr.write("\r" + line + padding)
            self.last_len = len(line)
            if completed == self.total:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()


def write_stats_tsv(path: Path, stats: Iterable[ChromStats]) -> None:
    rows = list(stats)
    header = list(ChromStats.__dataclass_fields__.keys())
    with path.open("wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for item in rows:
            values = []
            for key in header:
                value = getattr(item, key)
                if isinstance(value, float):
                    values.append(f"{value:.8g}")
                else:
                    values.append(str(value))
            handle.write("\t".join(values) + "\n")


def weighted_mean(values: Iterable[float], weights: Iterable[int]) -> float:
    v = np.asarray(list(values), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    return float((v * w).sum() / w.sum())


def write_report(path: Path, stats: list[ChromStats], plot_dir: Path, binary_dir: Path, meta: dict) -> None:
    total_valid = sum(item.valid_bases for item in stats)
    total_missing = sum(item.missing_bases for item in stats)
    total_size = sum(item.chrom_size for item in stats)
    mean_score = weighted_mean((item.mean for item in stats), (item.valid_bases for item in stats))
    median_values = sorted(stats, key=lambda item: item.median)
    positive_values = sorted(stats, key=lambda item: item.frac_positive, reverse=True)
    coverage_values = sorted(stats, key=lambda item: item.coverage_fraction, reverse=True)
    conserved_values = sorted(stats, key=lambda item: item.frac_ge_2, reverse=True)
    accelerated_values = sorted(stats, key=lambda item: item.frac_le_minus2, reverse=True)

    def top(items: list[ChromStats], attr: str, n: int = 5, pct: bool = False) -> str:
        pieces = []
        for item in items[:n]:
            value = getattr(item, attr)
            pieces.append(f"{item.chrom}={value:.2%}" if pct else f"{item.chrom}={value:.3f}")
        return ", ".join(pieces)

    with path.open("wt", encoding="utf-8") as handle:
        handle.write("# hg38 phyloP100way 位点进化保守性分数分布报告\n\n")
        handle.write("## 数据来源与编码\n\n")
        handle.write(f"- 数据目录：`{Path(meta.get('source_coordinate_index', '')).parent}`\n")
        handle.write(f"- binary 目录：`{binary_dir}`\n")
        handle.write(f"- track：`{meta.get('track', 'phyloP100way')}`\n")
        handle.write(f"- 编码：`real_score = encoded / {meta.get('scale')}`\n")
        handle.write(f"- 缺失值：`{meta.get('missing_value')}`，统计时已排除。\n")
        handle.write("- 解释：`phyloP > 0` 倾向保守，`phyloP < 0` 倾向加速演化，`0` 近似中性。\n\n")

        handle.write("## 总体结论\n\n")
        handle.write(f"- 统计染色体数：{len(stats)}\n")
        handle.write(f"- 总染色体长度：{total_size:,} bp\n")
        handle.write(f"- 有 phyloP 分数的位点：{total_valid:,} bp，占 {total_valid / total_size:.3%}\n")
        handle.write(f"- 缺失位点：{total_missing:,} bp，占 {total_missing / total_size:.3%}\n")
        handle.write(f"- 按有效位点加权的平均 phyloP：{mean_score:.4f}\n")
        handle.write(f"- 中位数最低的染色体：{top(median_values, 'median')}\n")
        handle.write(f"- 中位数最高的染色体：{top(list(reversed(median_values)), 'median')}\n")
        handle.write(f"- 正分数比例最高：{top(positive_values, 'frac_positive', pct=True)}\n")
        handle.write(f"- 覆盖率最高：{top(coverage_values, 'coverage_fraction', pct=True)}\n")
        handle.write(f"- 强保守比例 `phyloP >= 2` 最高：{top(conserved_values, 'frac_ge_2', pct=True)}\n")
        handle.write(f"- 加速演化比例 `phyloP <= -2` 最高：{top(accelerated_values, 'frac_le_minus2', pct=True)}\n\n")

        handle.write("## 解读要点\n\n")
        handle.write(
            "- 各染色体分布通常会在接近 0 的区域形成主峰，说明大量位点接近中性演化；右尾代表更强保守位点，左尾代表加速演化位点。\n"
        )
        handle.write(
            "- 中位数、IQR 和 P1/P99 用于观察主体分布位置和尾部宽度；`phyloP >= 2` 与 `phyloP <= -2` 比例用于粗略比较强保守/强加速位点负担。\n"
        )
        handle.write(
            "- 覆盖率差异反映 BED 覆盖区域和 UCSC phyloP track 缺失位点共同作用；缺失值没有参与均值、分位数或比例计算。\n\n"
        )

        handle.write("## 每条染色体统计表\n\n")
        handle.write(
            "| chrom | valid_bp | coverage | mean | std | median | IQR | P1 | P99 | >0 | >=2 | <=-2 | plot |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for item in sorted(stats, key=lambda x: natural_chrom_key(x.chrom)):
            try:
                png_rel = Path(item.png_path).relative_to(path.parent)
            except ValueError:
                png_rel = Path(item.png_path)
            handle.write(
                f"| {item.chrom} | {item.valid_bases:,} | {item.coverage_fraction:.2%} "
                f"| {item.mean:.3f} | {item.std:.3f} | {item.median:.3f} "
                f"| {item.q25:.3f}-{item.q75:.3f} | {item.q01:.3f} | {item.q99:.3f} "
                f"| {item.frac_positive:.2%} | {item.frac_ge_2:.2%} | {item.frac_le_minus2:.2%} "
                f"| [{item.chrom}]({png_rel}) |\n"
            )
        handle.write(f"\n完整 TSV：`{plot_dir / 'phyloP100way_chromosome_stats.tsv'}`\n")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    meta = read_meta(args.binary_dir)
    chroms = args.chrom if args.chrom else sorted(meta["files"], key=natural_chrom_key)
    missing = [chrom for chrom in chroms if chrom not in meta["files"]]
    if missing:
        raise ValueError(f"chromosomes absent from meta.json: {missing}")

    tasks = [
        {
            "chrom": chrom,
            "binary_dir": str(args.binary_dir),
            "rel_path": meta["files"][chrom],
            "chrom_size": int(meta["chrom_sizes"][chrom]),
            "scale": float(meta["scale"]),
            "missing_value": int(meta["missing_value"]),
            "chunk_size": args.chunk_size,
            "bins": args.bins,
            "out_dir": str(args.out_dir),
        }
        for chrom in chroms
    ]

    progress = ProgressReporter(len(tasks), args.progress_every, enabled=not args.no_progress_bar)
    progress.update(0, force=True)
    results: list[ChromStats] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_chromosome, task): task["chrom"] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            chrom = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"{chrom} failed: {exc}") from exc
            progress.update(len(results), force=len(results) == len(tasks))

    results.sort(key=lambda item: natural_chrom_key(item.chrom))
    stats_tsv = args.out_dir / "phyloP100way_chromosome_stats.tsv"
    write_stats_tsv(stats_tsv, results)
    write_report(args.report, results, args.out_dir, args.binary_dir, meta)

    print(f"plots_dir={args.out_dir}")
    print(f"stats_tsv={stats_tsv}")
    print(f"report_md={args.report}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
