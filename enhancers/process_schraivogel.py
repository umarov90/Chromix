import pandas as pd
import parse_data as parser
import numpy as np
import joblib
from main_params import MainParams
from liftover import get_lifter

p = MainParams()
train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
infos = train_info + valid_info + test_info

df_pos = pd.read_csv("data/enhancers/schraivogel_positive.tsv", sep="\t")
df_pos.drop_duplicates(inplace=True)
df_neg = pd.read_csv("data/enhancers/schraivogel_all.tsv", sep="\t")
df_neg.drop_duplicates(inplace=True)
# Drop rows from df_neg that have matching values in df
df_neg = df_neg.loc[~df_neg['enhancer'].isin(df_pos['enhancer'])]

def expand_column(df):
    # split the 'enhancer' column into three parts and convert to appropriate data types
    df[['chr', 'b', 'c']] = df['enhancer'].str.split(':|-', expand=True)
    df['b'] = df['b'].astype(int)
    df['c'] = df['c'].astype(int)
    df['mid'] = (df['b'] + df['c']) / 2
    df.drop('enhancer', axis=1, inplace=True)

expand_column(df_pos)
expand_column(df_neg)

a = df_pos['gene'].unique()
head = joblib.load(f"{p.pickle_folder}heads.gz")["hg38"]["expression"]
f5_tracks = []
for track in head:
    if "K562" in track:
        f5_tracks.append(track)
print(f"K562 tracks {len(f5_tracks)}")
load_info = []
gene_names = {}
for info in infos:
    ig = info[6]
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
    df_pos.loc[df_pos['gene'] == gene, 'gene'] = max_tss + 1  # not zero based!!!

def convert(df):
    converter = get_lifter('hg19', 'hg38')
    for index, row in df.iterrows():
        df.at[index, 'mid'] = converter[row["chr"]][row["mid"]][0][1]

df_pos = df_pos.rename(columns={'gene': 'tss'})
df_pos['Significant'] = True
convert(df_pos)
df_pos['tss'] = df_pos['tss'].astype(int)
convert(df_neg)

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
df_neg["tss"] = df_neg["tss"].astype(str)
df_neg = df_neg[df_neg['tss'].str.isdigit()]
df_neg["tss"] = df_neg["tss"].astype(int)
df_neg['Significant'] = False

df = pd.concat([df_pos, df_neg])
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True)

print(df.shape)
df = df[np.abs(df["mid"] - df["tss"]) > 2000]
df = df[np.abs(df["mid"] - df["tss"]) < 500000]
print(df.shape)
df = df[["chr", "tss", "mid", 'Significant']]
df.to_csv("data/enhancers/schraivogel_processed.tsv", index=False, sep="\t")