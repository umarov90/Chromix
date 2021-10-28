import pandas as pd
import numpy as np
import os

os.chdir(open("data_dir").read().strip())
df1 = pd.read_csv("cage_all.csv", delimiter=",", names=["pcc", "name"], header=None, index_col=False)
df2 = pd.read_csv("cage_good.csv", delimiter=",", names=["pcc", "name"], header=None, index_col=False)

pcc_all = []
pcc_good = []

for index, row in df2.iterrows():
    print(row)
    pcc_good.append(row["pcc"])
    pcc_all.append(df1[df1["name"] == row["name"]]["pcc"].values[0])

print(f"All: {np.mean(pcc_all)}, good only: {np.mean(pcc_good)}. Based on {len(pcc_good)} values.")