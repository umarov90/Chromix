tracks_folder = p.tracks_folder + "hg38/"
sizes = {}
full_names = {}
for filename in os.listdir(tracks_folder):
    if "FANTOM5" in filename:
        df = pd.read_csv(tracks_folder + filename, delim_whitespace=True, header=None, index_col=False)
        df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "score"})
        lib_size = df['score'].sum()
        sample_id = filename[filename.index("CNhs"):filename.index("CNhs") + 9]
        sizes[sample_id] = lib_size
        full_names[sample_id] = filename
df = pd.read_csv("data/ontology.csv", sep=",")
ftracks = {}
for i, row in df.iterrows():
    if not row["term"] in ftracks.keys():
        ftracks[row["term"]] = row["sample"]
    elif sizes[row["sample"]] > sizes[ftracks[row["term"]]]:
        ftracks[row["term"]] = row["sample"]
print(len(ftracks.values()))
for_cor = []
for_cor_inds = []
for val in ftracks.values():
    for_cor.append(full_names[val])
with open("fantom_tracks.tsv", 'w+') as f:
    f.write('\n'.join(for_cor))
for i, track in enumerate(heads["hg38"]["expression"]):
    if track in for_cor:
        for_cor_inds.append(str(i))
print(len(for_cor_inds))
with open("for_cor_inds.tsv", 'w+') as f:
    f.write('\n'.join(for_cor_inds))
exit()
cor_inds = pd.read_csv("data/for_cor_inds.tsv", sep="\t", header=None).iloc[:, 0].astype(int).tolist()