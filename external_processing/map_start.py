import os
from pathlib import Path
import pandas as pd
import sys

meta = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
meta = meta[meta["library_selection"] == "CAGE"]
meta.replace(r'^\s*$', "no_value", regex=True, inplace=True)
meta.fillna("no_value", inplace=True)
wd = os.path.dirname(os.path.abspath(sys.argv[1]))
sd = os.path.dirname(os.path.realpath(__file__))
jobs_num = 15
size = int(len(meta) / jobs_num)

list_of_dfs = [meta.loc[i:i+size-1,:] for i in range(0, len(meta), size)]

Path(wd + "/temp").mkdir(parents=True, exist_ok=True)
Path(wd + "/bed").mkdir(parents=True, exist_ok=True)
Path(wd + "/bam_to_ctss").mkdir(parents=True, exist_ok=True)
Path(wd + "/bam").mkdir(parents=True, exist_ok=True)
for i, df in enumerate(list_of_dfs):
    df.to_csv(wd + "/temp/" + str(i), sep='\t')

for i, df in enumerate(list_of_dfs):
    with open(f"{wd}/temp/job{i}", 'w+') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=map{i}\n")
        f.write(f"#SBATCH --output={wd}/temp/job{i}.out\n")
        f.write(f"#SBATCH --error={wd}/temp/job{i}.err\n")
        f.write(f"#SBATCH --partition=batch\n")
        f.write(f"python3 {sd}/map.py {wd}/temp/{i}")
    os.system(f"sbatch {wd}/temp/job{i}")
