from emma_3dgenome.io import load_contact_matrix
from emma_3dgenome.masks import detect_missing_bins


matrix = load_contact_matrix("sample.mcool", chrom="chr2", resolution=10000)
mask_info = detect_missing_bins(matrix, chrom="chr2", resolution=10000, mode="balanced")
mask_info.save("detected_out", chrom="chr2", resolution=10000)
