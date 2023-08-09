import pandas as pd
import parse_data as parser
import numpy as np
import joblib
from main_params import MainParams
from liftover import get_lifter

p = MainParams()
train_info, valid_info, test_info, _ = parser.parse_sequences(p)
infos = train_info + valid_info + test_info
df = pd.read_csv("data/validation/bingren2017_original.tsv", sep="\t")
# 31130734
        
df["mid"] = df["start"] + (df["end"] - df["start"]) // 2
converter = get_lifter('hg19', 'hg38')
inds = []
for index, row in df.iterrows():   
    try:
        df.at[index,'mid'] = converter[row["chr"]][row["mid"]][0][1]
    except:
        inds.append(index)

# Remove validation that could not be lifted
df.drop(df.index[inds], inplace=True)
df["tss"] = converter["chr6"][31130734][0][1]
df.to_csv("data/validation/bingren2017_processed.tsv", index=False, sep="\t")