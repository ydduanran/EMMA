# CLI

```bash
emma info sample.mcool --resolution 10000 --chrom chr2
emma detect sample.mcool --resolution 10000 --chrom chr2 --output detected_out/
emma restore sample.mcool --resolution 10000 --chrom chr2 --mask-regions missing.bed --output out/
emma reconstruct sample.mcool --resolution 10000 --chrom chr2 --output out/
```

`emma restore` requires one of `--mask`, `--mask-regions`, or `--auto-mask`.
