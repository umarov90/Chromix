import pandas as pd
import numpy as np
import os

os.chdir(open("data_dir").read().strip())
df1 = pd.read_csv("new_max.csv", delimiter=",", names=["pcc", "name"], header=None, index_col=False)

perf = {}
for index, row in df1.iterrows():
    mark = row["name"][row["name"].find(".")+len("."):row["name"].rfind(".pval")]
    perf.setdefault(mark, []).append(row["pcc"])

for key in perf.keys():
    print(f"{key}: {np.mean(perf[key])}, based on {len(perf[key])} values.")

# rerun model_all first!