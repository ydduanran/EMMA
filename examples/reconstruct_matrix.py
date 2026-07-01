from emma_3dgenome import EmmaRestorer
from emma_3dgenome.io import load_contact_matrix


matrix = load_contact_matrix("sample.mcool", chrom="chr2", resolution=10000)
restorer = EmmaRestorer(preset="default", device="cuda:0")
result = restorer.reconstruct(matrix, mode="conservative", blend=0.2)
result.save("emma_reconstruct_out")
