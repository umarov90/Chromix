import os
import pandas as pd
import pathlib
import joblib

script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])

dtypes = {"chr": str, "start": int, "end": int, "type": str}
df = pd.read_csv("data/enformer/data_human_sequences.bed", delim_whitespace=True, names=["chr", "start", "end", "type"],
                     dtype=dtypes, header=None, index_col=False)
df = df.loc[df['type'] == "test"]
df.to_csv('data/enformer/enformer_test.bed', sep='\t', index=False)
exit()
genome = joblib.load("pickle/genome.gz")

# df = df.loc[df['type'] == "test"]
for chr in genome.keys():
    if "_" in chr:
        continue
    c = len(df.loc[df['chr'] == chr])
    if c == 0:
        continue
    print(f"{chr} : {c}")
chosen_chr = "chr14"
df = df.loc[df['chr'] == chosen_chr]
df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) - 1
df.drop(['chr', 'start', 'end', 'type'], axis=1, inplace=True)
df = df.astype(int)
df.to_csv('data/enformer/chr1_test_val_points.tsv', sep='\t', index=False)


vals = df["mid"].tolist()
fasta = []
for i, v in enumerate(vals):
    s = f"> S {i}\n"
    s += genome[chosen_chr][v-10000:v+10001]
    # s += "\n"
    fasta.append(s)

with open("data/enformer/tss_test.fa", "w+") as text_file:
    text_file.write("\n".join(fasta))
