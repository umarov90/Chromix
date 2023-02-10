import pandas as pd
import parse_data as parser
import numpy as np
import joblib
from main_params import MainParams
from liftover import get_lifter

p = MainParams()
train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
infos = train_info + valid_info + test_info

df_pos = pd.read_csv("data/enhancers/gasperini_original.tsv", sep="\t")
df_neg = pd.read_csv("data/enhancers/gasperini_all_original.tsv", sep="\t")
# Drop rows from df_neg that have matching values in df
merged = df_neg.merge(df_pos, how='inner', on=['chr.candidate_enhancer', 'start.candidate_enhancer'])
df_neg.drop(merged.index, inplace=True)

# df = df.loc[df['high_confidence_subset'] == True]
a = df_pos['ENSG'].unique()
head = joblib.load(f"{p.pickle_folder}heads.gz")["hg38"]["expression"]
f5_tracks = []
for track in head:
    if "K562" in track:
        f5_tracks.append(track)
print(f"K562 tracks {len(f5_tracks)}")
load_info = []
gene_names = {}
for info in infos:
    ig = info[2]
    if "." in ig:
        ig = ig[:ig.index(".")]
    if ig in a:
        mid = info[1] // p.bin_size
        # len(load_info) is index to link info[1] (tss) to loaded values
        gene_names.setdefault(ig, []).append([len(load_info), info[1]])
        load_info.append([info[0], mid])

print(f"Load info {len(load_info)}") # num_genes x num_tss

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
    df_pos.loc[df_pos['ENSG'] == gene, 'ENSG'] = max_tss + 1  # not zero based!!!

def rename_convert(df):
    df.rename(columns={'chr.candidate_enhancer': 'chr'}, inplace=True)
    df["mid"] = df["start.candidate_enhancer"] + \
                (df["stop.candidate_enhancer"] - df["start.candidate_enhancer"]) // 2
    converter = get_lifter('hg19', 'hg38')
    for index, row in df.iterrows():
        df.at[index, 'mid'] = converter[row["chr"]][row["mid"]][0][1]
    df['mid'] = df['mid'].astype(int)

df_pos = df_pos.rename(columns={'ENSG': 'tss'})
df_pos['Significant'] = True
rename_convert(df_pos)
df_pos = df_pos[~df_pos['tss'].astype(str).str.startswith('E')]
df_pos['tss'] = df_pos['tss'].astype(int)
df_pos = df_pos[["chr", "tss", "mid", 'Significant']]
rename_convert(df_neg)

def get_closest_tss(row, df):
    filtered_df = df[(df['chr'] == row['chr']) &
      (abs(df['tss'] - row['mid']) > 2000) &
      (abs(df['tss'] - row['mid']) < 200000) &
      (abs(df['mid'] - row['mid']) > 2000)]
    if len(filtered_df) > 0:
        filtered_df['diff'] = abs(filtered_df['tss'] - row['mid'])
        closest_idx = filtered_df['diff'].idxmin()
        closest_tss = filtered_df.at[closest_idx, 'tss']
        return closest_tss
    return "E"

df_neg['tss'] = df_neg.apply(lambda row: get_closest_tss(row, df_pos), axis=1)
df_neg['Significant'] = False

df = pd.concat([df_pos, df_neg])
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True)

df = df[~df['tss'].astype(str).str.startswith('E')]
print(df.shape)
df = df[np.abs(df["mid"] - df["tss"]) > 2000]
print(df.shape)

df = df[["chr", "tss", "mid", 'Significant']]
df.to_csv("data/enhancers/gasperini_processed.tsv", index=False, sep="\t")