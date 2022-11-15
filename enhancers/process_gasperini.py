import pandas as pd
import parse_data as parser
import numpy as np
import joblib
from main_params import MainParams
from liftover import get_lifter

p = MainParams()
train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
infos = train_info + valid_info + test_info

df = pd.read_csv("data/gasperini.tsv", sep="\t")
df = df.loc[df['high_confidence_subset'] == True]
a = df['ENSG'].unique()
for i in range(len(a)):
	if "." in a[i]:
		a[i] = a[i][:a[i].index(".")]
		print(a[i])
head = joblib.load(f"{p.pickle_folder}heads.gz")["expression"]
f5_tracks = []
for track in head:
    if "FANTOM5" in track:
        f5_tracks.append(track)
print(f"FANTOM5 tracks {len(f5_tracks)}")
load_info = []
gene_names = {}
for info in infos:
    ig = info[2]
    if "." in ig:
        ig = ig[:ig.index(".")]
    if ig in a:
        mid = int(info[1] / p.bin_size)
        gene_names.setdefault(ig, []).append([len(load_info), info[1]])
        load_info.append([info[0], mid])

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
    df.loc[df['ENSG'] == gene, 'ENSG'] = max_tss
        
df["mid"] = df["start.candidate_enhancer"] + (df["stop.candidate_enhancer"] - df["start.candidate_enhancer"]) // 2
df = df[["ENSG", "chr.candidate_enhancer", "mid"]] 
df = df.rename(columns={'ENSG': 'Gene TSS', 'chr.candidate_enhancer': 'chr'})

converter = get_lifter('hg19', 'hg38')
for index, row in df.iterrows():   
    df.at[index,'mid'] = converter[row["chr"]][row["mid"]][0][1]

print(df.shape)
df = df[~df['Gene TSS'].astype(str).str.startswith('E')]
print(df.shape)

df.to_csv("data/gasperini_tss.tsv", index=False, sep="\t")