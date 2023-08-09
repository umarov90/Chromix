from liftover import get_lifter
import pandas as pd
import numpy as np
from main_params import MainParams

p = MainParams()
df_fulco = pd.read_csv("data/validation/fulco2019_original.tsv", sep="\t")
# Fulco2016 is processed separately
df_fulco = df_fulco[df_fulco.Reference != "Fulco2016"]
df_fulco = df_fulco.rename(columns={'Gene TSS': 'tss'})
df_fulco["mid"] = df_fulco["start"] + (df_fulco["end"] - df_fulco["start"]) // 2
df_fulco = df_fulco[["chr", "tss", "mid", 'Significant', "ABC Score"]]

converter = get_lifter('hg19', 'hg38')
for index, row in df_fulco.iterrows():   
    df_fulco.at[index,'mid'] = converter[row["chr"]][row["mid"]][0][1]
    df_fulco.at[index,'tss'] = converter[row["chr"]][row["tss"]][0][1]

df_fulco = df_fulco[np.abs(df_fulco["mid"] - df_fulco["tss"]) > 2000]
print(df_fulco['Significant'].value_counts())
df_fulco.to_csv("data/validation/fulco2019_processed.tsv", index=False, sep="\t")