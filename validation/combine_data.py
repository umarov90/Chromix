import pandas as pd
from main_params import MainParams

p = MainParams()

df_fulco2019 = pd.read_csv("data/validation/fulco2019_processed.tsv", sep="\t")
df_gasperini = pd.read_csv("data/validation/gasperini_processed.tsv", sep="\t")
df_fulco2016 = pd.read_csv("data/validation/fulco2016_processed.tsv", sep="\t")
df_schraivogel = pd.read_csv("data/validation/schraivogel_processed.tsv", sep="\t")

df = pd.concat([df_fulco2019, df_gasperini, df_fulco2016, df_schraivogel], ignore_index=True, axis=0)
df = df[["chr", "tss", "mid", 'Significant']]
df["tss"] = df["tss"].astype(int)
df["mid"] = df["mid"].astype(int)
print(df['Significant'].value_counts())
df.to_csv("data/validation/all.tsv", index=False, sep="\t")