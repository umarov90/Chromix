import pandas as pd
import re


def change_encode():
    a = pd.read_csv("/home/user/Desktop/encode.csv", header=None).to_numpy().flatten()
    b = []
    for l in a:
        new_track_name = ".".join(l.split(".")[:-3])
        new_track_name += ".parsed"
        b.append(new_track_name)
    with open("/home/user/Desktop/encode2.csv", "w") as text_file:
        text_file.write("\n".join(b))


def change_expression():
    a = pd.read_csv("/home/user/Desktop/expression.csv", header=None).to_numpy().flatten()
    b = []
    for l in a:
        s = re.split("_rep\d+(?:[^\.])*\.", l)
        new_track_name = s[0]
        new_track_name += ".parsed"
        b.append(new_track_name)

    with open("/home/user/Desktop/expression2.csv", "w") as text_file:
        text_file.write("\n".join(b))

# change_encode()
# change_expression()

df = pd.read_csv("/home/user/Desktop/all_track.metadata.tsv", sep="\t")
df['file_name'] = df['genome'] + '.' + df['file_name']
df = df.drop_duplicates()
df.to_csv("/home/user/Desktop/12all_track.metadata.tsv", index=False, sep="\t")