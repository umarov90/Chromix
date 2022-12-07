import pandas as pd
from main_params import MainParams

p = MainParams()

df_fulco2019 = pd.read_csv("data/enhancers/fulco2019_processed.tsv", sep="\t")
df_gasperini = pd.read_csv("data/enhancers/gasperini_processed.tsv", sep="\t")
df_fulco2016 = pd.read_csv("data/enhancers/fulco2016_processed.tsv", sep="\t")

df = pd.concat([df_fulco2019, df_gasperini, df_fulco2016], ignore_index=True, axis=0)
print(f"Fulco2019 {len(df_fulco2019)} Gasperini {len(df_gasperini)} Fulco2016 {len(df_fulco2016)}")
df = df[["chr", "tss", "mid", 'Significant']]
df["tss"] = df["tss"].astype(int)
df["mid"] = df["mid"].astype(int)
print(df['Significant'].value_counts())
df.to_csv("data/enhancers/all.tsv", index=False, sep="\t")