# CLI

```bash
emma info sample.mcool --resolution 10000 --chrom chr2
emma detect sample.mcool --resolution 10000 --chrom chr2 --output detected_out/
emma restore sample.mcool --resolution 10000 --chrom chr2 --mask-regions missing.bed --output out/
emma reconstruct sample.mcool --resolution 10000 --chrom chr2 --output out/
```

`emma restore` requires one of `--mask`, `--mask-regions`, or `--auto-mask`.

## Windowed runs

Use `--start-bin` and `--end-bin` to run EMMA on a selected local bin window:

```bash
emma restore sample.mcool \
  --resolution 10000 \
  --chrom chr2 \
  --start-bin 9500 \
  --end-bin 9700 \
  --auto-mask \
  --auto-mask-mode aggressive \
  --output out_chr2_9500_9700/
```

`--start-bin` is inclusive and `--end-bin` is exclusive. For `.cool` and `.mcool`, these bins are local to the selected chromosome. For `.npy` and `.npz`, they are matrix indices.
