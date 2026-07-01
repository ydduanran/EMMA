from emma_3dgenome import EmmaRestorer
from emma_3dgenome.io import load_contact_matrix
from emma_3dgenome.masks import load_mask_regions


matrix = load_contact_matrix("sample.mcool", chrom="chr2", resolution=10000)
mask_info = load_mask_regions("missing_regions.bed", chrom="chr2", resolution=10000, n_bins=matrix.shape[0])

restorer = EmmaRestorer(preset="default", device="cuda:0")
result = restorer.restore(matrix, mask=mask_info.mask, regions=mask_info.regions)
result.save("emma_out")
