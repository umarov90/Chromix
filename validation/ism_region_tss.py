import model as mo
import tensorflow as tf
from main_params import MainParams
import joblib
import numpy as np
import pandas as pd
import parse_data as parser
from pathlib import Path
from liftover import get_lifter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import RocCurveDisplay
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
import umap
import seaborn as sns
sns.set(font_scale = 1.0)

def get_signals(positions, head, kw, do_mean=True):
    signals = None
    for i, mark in enumerate(kw):
        chosen_tracks1 = []
        for track in head:
            # Change to correct cell type
            if mark in track.lower() and "ips" in track.lower():
                print(track.lower())
                chosen_tracks1.append(track)
        print(f"{len(chosen_tracks1)} {mark} tracks found")
        signal = parser.par_load_data(positions, chosen_tracks1, p)
        if do_mean:
            signal = np.mean(signal, axis=-1, keepdims=True)
        if signals is None:
            signals = signal
        else:
            signals = np.concatenate((signals, signal), axis=1)
        # print(f"{mark} shape is {signals.shape}")
    return signals

p = MainParams()
step = 1000
output_scores_info = []
heads = joblib.load(f"{p.pickle_folder}heads.gz")

inds = []
for i, track in enumerate(heads["hg38"]["expression"]):
    inds.append(i)

chosen_tracks1 = []
for i, track in enumerate(heads["hg38"]["expression"]):
    # Change to correct cell type
    if "K562" in track:
        chosen_tracks1.append(i)
    print(track)
print(chosen_tracks1)

df_targets = pd.read_csv("data/targets_human.txt", sep='\t')
chosen_tracks_enf = []
for i, row in df_targets.iterrows():
    # Change to correct cell type
    if "K562" in row["description"] and "CAGE" in row["description"]:
        chosen_tracks_enf.append(i)
        print( row["description"])
print(chosen_tracks_enf)
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
# df_tiling = pd.read_csv("data/validation/bingren2017_processed.tsv", sep="\t")
df_tiling = pd.read_csv("data/enhancers/fulco2016_tiling.tsv", sep="\t")
# MYC GATA1
df_tiling = df_tiling[df_tiling["chr"] == "chr8"]
# info = ["chr6", 31162957]
info = ["chr8", 128748314]
converter = get_lifter('hg19', 'hg38')
info[1] = converter[info[0]][info[1]][0][1]

our_model, head_inds = mo.prepare_model(p, heads)

seqs1 = []
seqs2 = []
eseqs1 = []
eseqs2 = []
mark_pos = []
seqs = []
start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
extra = start + p.input_size - len(one_hot[info[0]])
ns = one_hot[info[0]][start:start + p.input_size]
ns = ns[:, :-1]
gt_scores = []
for expression_region in range(start, start + 100000, step):
    mid = expression_region // p.bin_size
    mark_pos.append([info[0], mid])
    seq = ns.copy()
    s1 = expression_region - step
    s2 = expression_region + step
    sdf = df_tiling[(df_tiling['mid'] >= s1) & (df_tiling['mid'] <= s2)]
    if len(sdf) > 0:
        gt_scores.append(sdf["score"].mean())
    else:
        gt_scores.append(0)
    seq[expression_region - start - step: expression_region - start + step] = 0
    seqs1.append(ns.copy())
    seqs2.append(seq)
    eseqs1.append(np.concatenate([np.zeros((71616, 4)), ns.copy(), np.zeros((71616, 4))], axis=0))
    eseqs2.append(np.concatenate([np.zeros((71616, 4)), seq, np.zeros((71616, 4))], axis=0))
gt_scores = np.asarray(gt_scores)
gt_scores = -1 * gt_scores
gt_scores[gt_scores < 0] = 0

print(f"Number of sequences {len(seqs1)}")
marks = ["h3k4me1", "h3k4me3", "h3k27ac", "dnase"]
# marks = [ "h3k27ac"]
signals = get_signals(mark_pos, heads["hg38"]["epigenome"], marks)

effect_chromix, chromix_effects_h, max_change_chromix = mo.batch_predict_effect(p, our_model, np.asarray(seqs1), np.asarray(seqs2), chosen_tracks1)
joblib.dump((effect_chromix, chromix_effects_h, max_change_chromix), "chromix_myc_tiling.p", compress=3)
# effect_chromix, chromix_effects_h, max_change_chromix = joblib.load("chromix_effect.p")
import enformer_usage
effect_enformer, max_change_enformer = enformer_usage.calculate_effect(np.asarray(eseqs2), np.asarray(eseqs1), chosen_tracks_enf)
joblib.dump((effect_enformer, max_change_enformer), "enf_myc_tiling.p", compress=3)
# effect_enformer, max_change_enformer = joblib.load("enf_pouf1_tiling.p")

fig, axs = plt.subplots(len(marks) + 3,1,figsize=(12, 10), sharex=True)

for i, mark in enumerate(marks):
    val_arr = signals[:, i].flatten()
    x = range(len(val_arr))
    d1 = {'bin': x, 'val': val_arr}
    df1 = pd.DataFrame(d1)
    sns.lineplot(data=df1, x='bin', y='val', ax=axs[i])
    axs[i].fill_between(x, val_arr, alpha=0.5)
    axs[i].set_title(mark)

titles = ["CRISPRi", "Enformer", "Chromix"]
for i, arr in enumerate([gt_scores, max_change_enformer, max_change_chromix]):
    arr = arr.flatten()
    mid = len(arr) // 2
    arr[mid-5:mid+5]=0
    x = range(len(arr))
    d1 = {'bin': x, 'val': arr}
    df1 = pd.DataFrame(d1)
    sns.lineplot(data=df1, x='bin', y='val', ax=axs[len(marks) + i])
    axs[len(marks) + i].fill_between(x, arr, alpha=0.5)
    axs[len(marks) + i].set_title(titles[i])

plt.tight_layout()
plt.savefig(f"enhancer_scape/myc.pdf")