import pandas as pd
import parse_data as parser
import numpy as np
import joblib
from main_params import MainParams
from liftover import get_lifter

p = MainParams()
train_info, valid_info, test_info = parser.parse_sequences(p)
infos = train_info + valid_info + test_info
df = pd.read_csv("data/enhancers/fulco2016_original.tsv", sep="\t")
df.loc[df['Set'] == 'MYC Tiling', "Gene"] = "MYC"
df.loc[df['Set'] == 'GATA1 Tiling', "Gene"] = "GATA1"
a = df['Gene'].unique()
head = joblib.load(f"{p.pickle_folder}heads.gz")["expression"]
f5_tracks = []
for track in head:
    if "FANTOM5" in track and "K562" in track:
        f5_tracks.append(track)
print(f"FANTOM5 K562 tracks {len(f5_tracks)}")
load_info = []
gene_names = {}
found_genes = set()
for info in infos:
    ig = info[6]
    if ig in a:
        mid = info[1] // p.bin_size
        # len(load_info) is index to link info[1] (tss) to loaded values
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
# Major TSS is the one with the biggest average value in K562
for gene in gene_names:
    max_val = -1
    max_tss = -1
    for tss in gene_names[gene]:
        if gt[tss[0]] > max_val:
            max_val = gt[tss[0]]
            max_tss = tss[1]
    df.loc[df['Gene'] == gene, 'Gene'] = max_tss + 1 # not 0 based!
        
df["mid"] = df["start"] + (df["end"] - df["start"]) // 2
df = df.rename(columns={'Gene': 'tss', "CRISPRi Score": "score"})

# TSS is already hg38
converter = get_lifter('hg19', 'hg38')
inds = []
for index, row in df.iterrows():   
    try:
        df.at[index,'mid'] = converter[row["chr"]][row["mid"]][0][1]
    except:
        inds.append(index)
# Remove enhancers that could not be lifted
df.drop(df.index[inds], inplace=True)
df = df[np.abs(df["mid"] - df["tss"]) > 2000]
df_enhancer_screen_pos = df[df["score"] < -0.5].copy()
df_enhancer_screen_pos['Significant'] = True

df_enhancer_screen_neg = df[df["score"] > 0.0].copy()
df_enhancer_screen_neg['Significant'] = False

df = pd.concat([df_enhancer_screen_pos, df_enhancer_screen_neg], ignore_index=True, axis=0)
print(df['Significant'].value_counts())

df1 = df[df.Set == "Protein Coding Gene Promoters"]
df1 = df1[["chr", "tss", "mid", 'Significant']]
df1.to_csv("data/enhancers/fulco2016_processed.tsv", index=False, sep="\t")

df2 = df[df['Set'].isin(["MYC Tiling", "GATA1 Tiling"])]
df2 = df2[["chr", "tss", "mid", 'Significant']]
df2.to_csv("data/enhancers/fulco2016_tiling.tsv", index=False, sep="\t")