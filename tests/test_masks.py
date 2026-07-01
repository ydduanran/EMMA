from pathlib import Path

import numpy as np

from emma_3dgenome.masks import filter_bins_by_exclude_bed, mask_from_bin_regions, mask_from_bins, merge_bins_to_regions


def test_mask_from_bins_shape_and_rows_cols():
    mask = mask_from_bins(10, [2, 5])
    assert mask.shape == (10, 10)
    assert mask[2, :].all()
    assert mask[:, 5].all()
    assert not mask[0, 1]


def test_mask_from_bin_regions():
    mask = mask_from_bin_regions(12, [(3, 5)])
    assert mask[3, :].all()
    assert mask[4, :].all()
    assert mask[:, 3].all()
    assert not mask[2, 2]


def test_merge_bins_to_regions():
    assert merge_bins_to_regions([1, 2, 3, 7, 8], min_region_len=1, merge_gap=0) == [(1, 4), (7, 9)]
    assert merge_bins_to_regions([1, 3, 4], min_region_len=1, merge_gap=1) == [(1, 5)]


def test_exclude_bed_filtering(tmp_path: Path):
    bed = tmp_path / "exclude.bed"
    bed.write_text("chr1\t20\t40\n", encoding="utf-8")
    kept, excluded = filter_bins_by_exclude_bed([1, 2, 3, 4], chrom="chr1", resolution=10, exclude_bed=bed)
    assert kept == [1, 4]
    assert excluded == [2, 3]
