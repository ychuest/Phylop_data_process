# hg38 phyloP100way conservation score processing

This repository contains the code used to generate chromosome-level dense
`int16` arrays for the UCSC hg38 100-way phyloP conservation track.

The generated data are not intended to be committed to GitHub. The local
generated dataset is:

```text
/data/huda01/yangcheng/projects/work2/data/hg38/hg38_conservation_scores/binary/phyloP100way
```

## What Was Generated

Final binary output:

```text
hg38_conservation_scores/binary/phyloP100way/
  meta.json
  chr1.i16.npy
  chr2.i16.npy
  ...
  chr22.i16.npy
  chrX.i16.npy
```

Encoding:

```text
real_phyloP = encoded_int16 / 1000.0
missing_value = -32768
dtype = int16
```

Local production status:

- Input BED intervals: 38,171
- BED split counts: train 34,021; valid 2,213; test 1,937
- Input BED interval length: 131,072 bp each
- Merged non-overlapping regions: 416
- Download windows after 1 Mb chunking: 2,953
- Requested unique bp: 2,727,870,737
- Covered bp in bedGraph outputs: 2,705,621,524
- Download/verification status: 2,953 `verified_existing`
- Binary chromosomes: 23
- Total hg38 chromosome length represented in `meta.json`: 3,031,042,417 bp

## Raw Inputs

Required local inputs:

```text
human-sequences.bed
hg38.ml.fa.fai
```

In the original local run these were:

```text
/data/huda01/yangcheng/projects/work2/data/hg38/human-sequences.bed
/data/huda01/yangcheng/projects/work2/data/hg38/hg38.ml.fa.fai
```

External UCSC source track:

```text
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw
```

External extraction tool:

```text
bigWigToBedGraph
```

The downloader can use a system `bigWigToBedGraph`, a user-supplied
`--tool`, or download a UCSC Linux binary into `hg38_conservation_scores/tools/`.

## Processing Pipeline

1. Read `human-sequences.bed` and `hg38.ml.fa.fai`.
2. Validate BED coordinates against chromosome sizes from the FAI.
3. Merge overlapping BED intervals to avoid downloading the same base more than once.
4. Split merged regions into non-overlapping chunks of at most 1,000,000 bp.
5. Use `bigWigToBedGraph` to extract UCSC `phyloP100way` values for each chunk.
6. Validate every bedGraph row for chromosome, coordinate bounds, row order, and finite score.
7. Write:
   - `merged_intervals.bed`
   - `download_windows.bed`
   - `interval_to_download_windows.tsv`
   - `manifest.tsv`
   - `coordinate_index.tsv`
   - `phyloP100way/bedGraph/*.bedGraph.gz`
8. Convert bedGraph chunks into one dense `int16 .npy` array per chromosome.
9. Validate the binary arrays against the source bedGraph chunks.
10. Optionally plot per-chromosome score distributions.

## Scripts

```text
download_hg38_conservation.py       # remote BigWig interval extraction to bedGraph
build_conservation_memmap.py        # bedGraph -> chromosome-level int16 .npy arrays
validate_conservation_memmap.py     # sampled or exhaustive bedGraph-vs-memmap validation
plot_phyloP_distribution.py         # distribution plots and Markdown summary
```

## Reproduction Commands

Run from this repository directory, or pass absolute paths explicitly.

### 1. Extract bedGraph Chunks

```bash
python download_hg38_conservation.py \
  --bed /path/to/human-sequences.bed \
  --fai /path/to/hg38.ml.fa.fai \
  --out-dir hg38_conservation_scores \
  --track phyloP100way \
  --workers 24 \
  --progress-every 25 \
  --timeout 1800 \
  --retries 5
```

For a dry run:

```bash
python download_hg38_conservation.py \
  --bed /path/to/human-sequences.bed \
  --fai /path/to/hg38.ml.fa.fai \
  --out-dir hg38_conservation_scores \
  --track phyloP100way \
  --dry-run
```

To verify existing bedGraph files and rewrite manifests:

```bash
python download_hg38_conservation.py \
  --bed /path/to/human-sequences.bed \
  --fai /path/to/hg38.ml.fa.fai \
  --out-dir hg38_conservation_scores \
  --track phyloP100way \
  --verify-only
```

### 2. Build Dense Binary Arrays

```bash
python build_conservation_memmap.py \
  --scores-dir hg38_conservation_scores \
  --fai /path/to/hg38.ml.fa.fai \
  --track phyloP100way \
  --scale 1000.0 \
  --missing-value -32768 \
  --force
```

### 3. Validate Binary Arrays

Exhaustive validation:

```bash
python validate_conservation_memmap.py \
  --scores-dir hg38_conservation_scores \
  --track phyloP100way \
  --files 0 \
  --rows-per-file 0 \
  --workers 24 \
  --progress-every 5
```

Fast sampled validation:

```bash
python validate_conservation_memmap.py \
  --scores-dir hg38_conservation_scores \
  --track phyloP100way \
  --files 50 \
  --rows-per-file 2000 \
  --workers 8
```

### 4. Plot Distribution Summary

```bash
python plot_phyloP_distribution.py \
  --binary-dir hg38_conservation_scores/binary/phyloP100way \
  --out-dir phyloP100way_distribution_plots \
  --report phyloP100way_distribution_report.md \
  --workers 23 \
  --chunk-size 10000000 \
  --progress-every 1
```

## Expected Intermediate Files

```text
hg38_conservation_scores/
  merged_intervals.bed
  download_windows.bed
  interval_to_download_windows.tsv
  manifest.tsv
  coordinate_index.tsv
  phyloP100way/bedGraph/*.bedGraph.gz
  binary/phyloP100way/*.i16.npy
  binary/phyloP100way/meta.json
```

`manifest.tsv` is the download-level audit table. `coordinate_index.tsv` is the
minimal index used by `build_conservation_memmap.py`.

## Notes and Limitations

- The scripts use half-open 0-based genomic intervals, matching BED and
  bedGraph conventions.
- Missing bases are represented by `-32768`; this value is excluded from
  distribution summaries and should be treated as unavailable conservation
  evidence during training.
- The pipeline extracts only regions covered by `human-sequences.bed` after
  merging, not the entire UCSC genome-wide BigWig into bedGraph.
- The final `.npy` arrays are dense across full chromosome coordinates. Bases
  outside downloaded/covered bedGraph rows remain `missing_value`.
- Large data files, UCSC cache files, downloaded binaries, bedGraph files, and
  `.npy` arrays should not be committed.

