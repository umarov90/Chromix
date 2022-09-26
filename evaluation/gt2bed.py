from main_params import MainParams
import joblib
import numpy as np
import pandas as pd
import parse_data as parser
from pathlib import Path
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=2.0)
p = MainParams()
heads = joblib.load(f"{p.pickle_folder}heads.gz")

# output_expression = parser.par_load_data(output_scores_info, heads["expression"], p)
# output_sc = parser.par_load_data(output_scores_info, heads["sc"], p)
# for track in heads["sc"]:
#     if "astrocyte" not in track.lower():
#         continue
#     print(track)
#     parsed_track0 = joblib.load(p.parsed_tracks_folder + track)
#     break

# for track in heads["expression"]:
#     if "astrocyte" not in track.lower():
#         continue
#     print(track)
#     parsed_track2 = joblib.load(p.parsed_tracks_folder + track)
#     break

# pc = stats.pearsonr(parsed_track0["chr1"], parsed_track2["chr1"])[0]
# sc = stats.spearmanr(parsed_track0["chr1"], parsed_track2["chr1"])[0]
# print(f"{pc} {sc}")
# train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
# v1 = []
# v2 = []
# for i, info in enumerate(valid_info):
#     t1 = parsed_track0[info[0]][info[1] // p.bin_size]
#     t2 = parsed_track2[info[0]][info[1] // p.bin_size]
#     # if t1 > 0 and t2 > 0:
#     v1.append(t1)
#     v2.append(t2)

# np.savetxt('v1.csv', v1, delimiter=',')
# np.savetxt('v2.csv', v2, delimiter=',')

# df1 = pd.DataFrame(v1, columns=['val'])
# df1['Type'] = 'scEnd5'
# df2 = pd.DataFrame(v2, columns=['val'])
# df2['Type'] = 'FANTOM5'
# merged = pd.concat([df1,df2], ignore_index=True, axis=0)
# pal = sns.cubehelix_palette(2, rot=-.25, light=.7)
# g = sns.FacetGrid(merged, row="Type", hue="Type", aspect=10, height=2, palette=pal)

# g.map(sns.kdeplot, "val", bw_adjust=.6, cut=5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
# g.map(sns.kdeplot, "val", bw_adjust=.6, cut=5, clip_on=False, color="w", lw=2)
# g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(0, .1, label, fontweight="bold", color=color,
#             ha="left", va="center", transform=ax.transAxes)

# g.map(label, "Type")
# g.fig.subplots_adjust(hspace=-.7)
# g.set(yticks=[], xlabel="val", ylabel="", xlim=(0.2, 1), title="")
# g.despine(bottom=True, left=True)

# plt.savefig("hist1.png")

# np.count_nonzero(np.asarray(v1)==0)

# fig, ax = plt.subplots(figsize=(6, 6))
# sns.regplot(x=v1, y=v2)
# ax.set(xlabel='scEnd5', ylabel='FANTOM5')
# fig.tight_layout()
# plt.savefig(f"1.png")
# plt.close(fig)

# pc = stats.pearsonr(v1, v2)[0]
# sc = stats.spearmanr(v1, v2)[0]
# print(f"{pc} {sc}")
for n, track in enumerate(heads["expression"]):
    # print(track)
    # if "scEnd5" not in track:
    #     continue
    if "CNhs12119" not in track:
        continue
    # if "FANTOM5" not in track or "response" in track:
    #     continue
    parsed_track = joblib.load(p.parsed_tracks_folder + track)
    with open("bed_output/f5_" + track + ".bedGraph", 'w+') as f:
        for i, val in enumerate(parsed_track["chr1"]):
            f.write(f"chr1\t{i * p.bin_size}\t{(i + 1) * p.bin_size}\t{val}")
            f.write("\n")
    break
    if n > 5:
        break
