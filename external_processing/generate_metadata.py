import os
import pandas as pd
import sys


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


def find_between_r(s, first, last):
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


df = pd.read_csv(sys.argv[1], sep="\t", header=None, names=["path"], index_col=False)
wd = os.path.dirname(os.path.abspath(sys.argv[1]))

row_ids = []
descriptions = []
for index, row in df.iterrows():
    path = row["path"]
    row_ids.append("CNhs" + find_between(path, ".CNhs", "."))
    descriptions.append(find_between_r(path, "/", ".CNhs").replace("_", " "))

df["id"] = row_ids
df["description"] = descriptions
df = df[["id", "path", "description"]]
df.to_csv(f"{wd}/metadata.tsv", sep='\t', index=False, header=None)
