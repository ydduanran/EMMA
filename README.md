# EMMA

EMMA is an EMD-guided restoration toolkit for chromatin interaction maps. It restores complete or low-quality genomic-bin regions in Hi-C, Pore-C, and contact-like 3D genome matrices by combining distance-diagonal signal decomposition, masked IMF autoencoder correction, and mode-weighted reconstruction.

![EMMA overview](assets/emma_overview.png)

## What EMMA Does

EMMA supports three common workflows:

- Restore known missing regions from a user-provided mask.
- Automatically detect low-coverage or missing genomic bins, then restore them.
- Reconstruct or lightly enhance a contact matrix without an explicit missing mask.

The restore mode keeps observed entries unchanged and only replaces entries marked by the imputation mask.

## Installation

### Option A. Install From PyPI

After the package is released to PyPI:

```bash
pip install emma-3dgenome
```

### Option B. Install Directly From GitHub

```bash
pip install "git+https://github.com/ydduanran/EMMA.git"
```

### Option C. Install From Local Source

Clone the repository and install it locally:

```bash
git clone https://github.com/ydduanran/EMMA.git
cd EMMA
pip install .
```

For editable development:

```bash
pip install -e ".[dev]"
```

If your machine has no internet access but already has the required build tools installed, use:

```bash
pip install -e . --no-build-isolation
```

Core dependencies include `numpy`, `scipy`, `torch`, `cooler`, `EMD-signal`, `scikit-learn`, and `scikit-image`.

## Version 0.2.0 Performance Notes

EMMA 0.2.0 changes the default masked-autoencoder training path to avoid recomputing PyEMD inside every DataLoader sample. The default training pipeline now pseudo-masks IMF channels directly and keeps EMD decomposition in the matrix-level preprocessing stage. This makes GPU training substantially less CPU-bound.

To reproduce the slower 0.1.x training behavior that recomputes EMD for every pseudo-masked sample, use:

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --mask-regions missing_regions.bed \
  --recompute-pseudo-emd \
  --output emma_out_legacy/
```

For CUDA runs on larger windows, start with:

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --mask-regions missing_regions.bed \
  --device cuda:0 \
  --batch-size 256 \
  --num-workers 8 \
  --inference-batch-size 256 \
  --output emma_out/
```

## Version 0.2.1 Windowed CLI

EMMA supports direct bin-window analysis from the command line. `--start-bin` is inclusive and `--end-bin` is exclusive, using local bins within the selected chromosome or matrix.

```bash
emma restore /mnt/disk1/duanran/emhic_baseline/data/H1_ESC_4DNFI82R42AD.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --start-bin 9500 \
  --end-bin 9700 \
  --auto-mask \
  --auto-mask-mode aggressive \
  --device cuda:0 \
  --preset default \
  --max-diag 150 \
  --patch-len 64 \
  --stride 16 \
  --epochs 35 \
  --n-samples 30000 \
  --batch-size 128 \
  --recompute-pseudo-emd \
  --output emma_chr2_9500_9700_out/
```

When writing detected BED regions for windowed `.cool` or `.mcool` input, EMMA adds `--start-bin` back to the local detected bins so the BED coordinates remain chromosome-level genomic coordinates.

## Notebooks

The repository includes two executed notebooks under [`notebooks/`](notebooks/). Outputs are kept in the notebooks so readers can inspect representative results without rerunning the full workflows.

- [`EMMA_tutorial.ipynb`](notebooks/EMMA_tutorial.ipynb): a step-by-step tutorial for loading an `.mcool` Hi-C window, detecting real missing bins, running EMMA restoration, and visualizing the restored matrix.
- [`EMMA_baseline_analysis.ipynb`](notebooks/EMMA_baseline_analysis.ipynb): the baseline-comparison analysis pipeline for reproducing the paper-style comparison tables, including baseline comparisons and EMMA ablation summaries.

## Quick Start

### 1. Restore With A BED Missing-Region File

Use this when you know which genomic intervals need imputation.

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --mask-regions missing_regions.bed \
  --output emma_out/
```

`missing_regions.bed` should use:

```text
chrom  start  end
```

Example:

```text
chr2    2000000    2250000
chr2    7600000    7900000
```

### 2. Restore With A Boolean Matrix Mask

Use this when you already have a square boolean mask where `True` marks entries to impute.

```bash
emma restore sample.npy \
  --mask mask.npy \
  --output emma_out/
```

### 3. Auto-Detect Missing Bins And Restore

Use this when missing or low-coverage bins are not known in advance.

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --auto-mask \
  --auto-mask-mode balanced \
  --output emma_auto_out/
```

Available auto-mask modes:

- `conservative`
- `balanced`
- `aggressive`

You can exclude assembly gaps, centromeres, telomeres, blacklist regions, or other regions that should not be imputed:

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --auto-mask \
  --exclude-bed hg38_exclude_regions.bed \
  --output emma_auto_out/
```

### 4. Reconstruct Without Explicit Missing Regions

Use this for conservative EMMA-style matrix reconstruction or enhancement.

```bash
emma reconstruct sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --mode conservative \
  --blend 0.2 \
  --output reconstructed_out/
```

Modes:

- `conservative`: lightly blends the reconstruction with the original matrix.
- `full`: uses the reconstructed matrix directly.

## Input Formats

EMMA currently supports:

- `.cool`
- `.mcool`
- `.npy`
- `.npz`

Rules:

- `.mcool` requires `--resolution`.
- `.cool` and `.mcool` require `--chrom`.
- `.npy` should contain a square contact matrix.
- `.npz` reads key `matrix` if present; otherwise it reads the first array. Use `--key` to choose a specific array.

## Output Files

`emma restore` writes:

```text
restored.npy
prediction_only.npy
masked_input.npy
mask.npy
mask_regions.bed
config.json
report.json
diag_stats.json
log.txt
```

`emma detect` writes:

```text
mask.npy
detected_missing_bins.tsv
detected_missing_regions.bed
excluded_bins.tsv
auto_mask_diagnostics.tsv
report.json
```

`emma reconstruct` writes:

```text
reconstructed.npy
difference.npy
config.json
report.json
diag_stats.json
log.txt
```

## Python API

```python
from emma_3dgenome import EmmaRestorer
from emma_3dgenome.io import load_contact_matrix
from emma_3dgenome.masks import load_mask_regions

matrix = load_contact_matrix(
    "sample.mcool",
    chrom="chr2",
    resolution=10000,
)

mask_info = load_mask_regions(
    "missing_regions.bed",
    chrom="chr2",
    resolution=10000,
    n_bins=matrix.shape[0],
)

restorer = EmmaRestorer(preset="default", device="cuda:0")
result = restorer.restore(
    matrix,
    mask=mask_info.mask,
    regions=mask_info.regions,
)
result.save("emma_out")
```

Auto-mask restoration:

```python
from emma_3dgenome import EmmaRestorer
from emma_3dgenome.io import load_contact_matrix

matrix = load_contact_matrix("sample.mcool", chrom="chr2", resolution=10000)
restorer = EmmaRestorer(preset="default", device="cuda:0")

result = restorer.restore_auto(
    matrix,
    chrom="chr2",
    resolution=10000,
    auto_mask_mode="balanced",
)
result.save("emma_auto_out")
```

Matrix reconstruction:

```python
result = restorer.reconstruct(matrix, mode="conservative", blend=0.2)
result.save("reconstructed_out")
```

## IMF Parameters

Default mode-weighted reconstruction parameters:

```text
max_imfs = 5
imf_weights = 0.08 1.35 1.20 1.90 0.80
residual_weight = 1.0
diag_calib_strength = 0.20
```

Interpretation:

- IMF1: high-frequency noise component, strongly down-weighted.
- IMF2: local structure component, enhanced.
- IMF3: intermediate-scale structure component, enhanced.
- IMF4: domain or boundary-related structure component, strongly enhanced.
- IMF5: low-frequency structure component, slightly retained.
- Residual: global trend, retained.

Override from CLI:

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --mask-regions missing_regions.bed \
  --max-imfs 5 \
  --imf-weights 0.08 1.35 1.20 1.90 0.80 \
  --residual-weight 1.0 \
  --diag-calib-strength 0.20 \
  --output emma_out/
```

## Presets

Available presets:

- `default`
- `paper`
- `smooth`
- `sharp`
- `conservative`
- `fast`

Use `fast` for small smoke tests. Use `default` or `paper` for standard restoration.

## Minimal Test

```bash
python -m compileall -q src tests
python -m pytest -q tests
```

If `pytest` is not installed:

```bash
pip install -e ".[dev]"
```

## Citation

If you use EMMA in your work, please cite the EMMA manuscript when it becomes available.

```bibtex
@article{emma2026,
  title = {EMMA: EMD-guided masked autoencoder restoration of chromatin interaction maps},
  author = {To be updated},
  journal = {To be updated},
  year = {2026}
}
```

## License

This project is released under the MIT License.
