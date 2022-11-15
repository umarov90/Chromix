import pandas as pd
import parse_data as parser
import numpy as np
import joblib
from main_params import MainParams
from liftover import get_lifter

p = MainParams()
train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
infos = train_info + valid_info + test_info

df = pd.read_csv("data/enhancer_screen.tsv", sep="\t")
a = df['Gene'].unique()
head = joblib.load(f"{p.pickle_folder}heads.gz")["expression"]
f5_tracks = []
for track in head:
    if "FANTOM5" in track:
        f5_tracks.append(track)
print(f"FANTOM5 tracks {len(f5_tracks)}")
load_info = []
gene_names = {}
found_genes = set()
for info in infos:
    ig = info[6]
    if ig in a:
        mid = int(info[1] / p.bin_size)
        gene_names.setdefault(ig, []).append([len(load_info), info[1]])
        load_info.append([info[0], mid])
        found_genes.add(ig)

not_found_genes = set(a) - found_genes
print(f"Not found {len(not_found_genes)} genes")
print(df.shape)
df = df[~df.Gene.isin(not_found_genes)]
print(df.shape)

print(f"Load info {len(load_info)}")        

print("Loading ground truth tracks")
gt = parser.par_load_data(load_info, f5_tracks, p)
print(gt.shape)
gt = np.mean(gt, axis=-1)
print(gt.shape)

for gene in gene_names:
    max_val = -1
    max_tss = -1
    for tss in gene_names[gene]:
        if gt[tss[0]] > max_val:
            max_val = gt[tss[0]]
            max_tss = tss[1]
    df.loc[df['Gene'] == gene, 'Gene'] = max_tss
        
df["mid"] = df["start"] + (df["end"] - df["start"]) // 2
df = df[["Gene", "chr", "mid", "CRISPRi Score"]] 
df = df.rename(columns={'Gene': 'Gene TSS', "CRISPRi Score" : "score"})

# TSS is already hg38
converter = get_lifter('hg19', 'hg38')
for index, row in df.iterrows():   
    df.at[index,'mid'] = converter[row["chr"]][row["mid"]][0][1]

df.to_csv("data/enhancer_screen_tss.tsv", index=False, sep="\t")