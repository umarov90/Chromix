from liftover import get_lifter
import pandas as pd
import numpy as np
from main_params import MainParams

p = MainParams()

df_fulco = pd.read_csv("data/enhancers/fulco.tsv", sep="\t")
df_fulco["mid"] = df_fulco["start"] + (df_fulco["end"] - df_fulco["start"]) // 2
converter = get_lifter('hg19', 'hg38')
for index, row in df_fulco.iterrows():   
    df_fulco.at[index,'mid'] = converter[row["chr"]][row["mid"]][0][1]
    df_fulco.at[index,'Gene TSS'] = converter[row["chr"]][row["Gene TSS"]][0][1]

print("df_fulco")
print(df_fulco['Significant'].value_counts())

df_gasperini = pd.read_csv("data/enhancers/gasperini_tss.tsv", sep="\t")
df_gasperini['Significant'] = True
print("df_gasperini")
print(df_gasperini['Significant'].value_counts())

df_enhancer_screen = pd.read_csv("data/enhancers/enhancer_screen_tss.tsv", sep="\t")

df_enhancer_screen_pos =  df_enhancer_screen[df_enhancer_screen["score"]  < -1.0].copy()
df_enhancer_screen_pos['Significant'] = True

df_enhancer_screen_neg =  df_enhancer_screen[df_enhancer_screen["score"]  > 0.0].copy()
df_enhancer_screen_neg['Significant'] = False

df_enhancer_screen = pd.concat([df_enhancer_screen_pos, df_enhancer_screen_neg], ignore_index=True, axis=0)
print("df_enhancer_screen")
print(df_enhancer_screen['Significant'].value_counts())

df = pd.concat([df_fulco, df_gasperini, df_enhancer_screen], ignore_index=True, axis=0)
print(f"Fulco {len(df_fulco)} Gasperini {len(df_gasperini)} Enhancer screen {len(df_enhancer_screen)}")
# df = df_gasperini
# df = df_fulco
df["Gene TSS"] = df["Gene TSS"].astype(int)
df = df[["chr", "Gene TSS", "mid", 'Significant']] 

print("df")
print(df['Significant'].value_counts())
df.to_csv("data/enhancers/all.tsv", index=False, sep="\t")