import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=2.0)
# sns.set(font_scale = 2)
os.chdir("/home/user/data")

DNase = pd.read_csv("all_track_spearman_DNase_tss.csv", sep=",", names=['Correlation', 'Type'])
DNase.iloc[:, 1] = "DNase"
ATAC = pd.read_csv("all_track_spearman_ATAC_tss.csv", sep=",", names=['Correlation', 'Type'])
ATAC.iloc[:, 1] = "ATAC"
CAGE = pd.read_csv("all_track_spearman_CAGE_tss.csv", sep=",", names=['Correlation', 'Type'])
CAGE.iloc[:, 1] = "CAGE"
Histone_ChIP = pd.read_csv("all_track_spearman_Histone_ChIP_tss.csv", sep=",", names=['Correlation', 'Type'])
Histone_ChIP.iloc[:, 1] = "Histone_ChIP"
TF_ChIP = pd.read_csv("all_track_spearman_TF_ChIP_tss.csv", sep=",", names=['Correlation', 'Type'])
TF_ChIP.iloc[:, 1] = "TF_ChIP"
all_df = [DNase, ATAC, CAGE, Histone_ChIP, TF_ChIP]
merged = pd.concat(all_df, ignore_index=True, axis=0)
merged.reset_index(drop=True, inplace=True)
# sns.histplot(data=merged, x="Correlation", hue="Type")

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(all_df), rot=-.25, light=.7)
g = sns.FacetGrid(merged, row="Type", hue="Type", aspect=10, height=2, palette=pal)

g.map(sns.kdeplot, "Correlation", bw_adjust=.6, cut=5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Correlation", bw_adjust=.6, cut=5, clip_on=False, color="w", lw=2)
g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .1, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "Type")
g.fig.subplots_adjust(hspace=-.7)
g.set(yticks=[], xlabel="Correlation", ylabel="", xlim=(0.2, 1), title="")
g.despine(bottom=True, left=True)

plt.savefig("figures/hist.png")