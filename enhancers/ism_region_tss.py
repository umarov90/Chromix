import model as mo
import tensorflow as tf
from main_params import MainParams
import joblib
import numpy as np
import pandas as pd
import parse_data as parser
from pathlib import Path
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
sns.set(font_scale = 2.5)

def get_signals(mark_positions):
    signals = None
    for i, mark in enumerate(marks):
        chosen_tracks1 = []
        for track in head["epigenome"]:
            # if "K562" in track:
            if mark in track.lower():
                chosen_tracks1.append(track)
        # print(f"{len(chosen_tracks1)} {mark} tracks found")
        signal = parser.par_load_data_temp(mark_positions, chosen_tracks1, p)
        signal = np.mean(signal, axis=-1, keepdims=True)
        if signals is None:
            signals = signal
        else:
            signals = np.concatenate((signals, signal), axis=-1)
    return signals

p = MainParams()


one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
step = 640
output_scores_info = []
head = joblib.load(f"{p.pickle_folder}heads.gz")["hg38"]

inds = []
for i, track in enumerate(head["expression"]):
    inds.append(i)

train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
infos = train_info + valid_info + test_info
# infos = []
# for info in all_infos:
#     if info[0] == "chr11":
#         infos.append(info)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head["expression"])
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))

clf = joblib.load("RF.pkl")
pca = joblib.load("PCA.pkl")
gene_enhancers = {}
for i, info in enumerate(infos):
    if "irf8" not in info[6].lower():
        continue
    seqs1 = []
    seqs2 = []
    distances = []
    mark_pos = []
    seqs = []
    start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
    extra = start + p.input_size - len(one_hot[info[0]])
    if start < 0:
        ns = one_hot[info[0]][0:start + p.input_size]
        ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
    elif extra > 0:
        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
        ns = np.concatenate((ns, np.zeros((extra, 5))))
    else:
        ns = one_hot[info[0]][start:start + p.input_size]
    ns = ns[:, :-1]
    for expression_region in range(start, start + p.input_size, step):
        distances.append(abs(expression_region - info[1]))
        mid = expression_region // p.bin_size
        mark_pos.append([info[0], mid])
        seq = ns.copy()
        seq[expression_region - start: expression_region - start + step] = 0
        seqs1.append(ns.copy())
        seqs2.append(seq)
    
    print(f"Number of sequences {len(seqs1)}")
    # marks = ["h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3", "h3k36me3", "h3k27ac"]
    # signals = get_signals(mark_pos)

    _, fold_changes = mo.batch_predict_effect(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
    dif = mo.batch_predict_effect_x(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
    joblib.dump([dif, fold_changes], "irf8.p", compress=3)
    # dump = joblib.load("irf8.p")
    # dif = dump[0]
    # fold_changes = dump[1]
    distances = np.expand_dims(np.asarray(distances), axis=1)
    add_features = distances # np.concatenate((signals, distances, fold_changes), axis=-1)
    dif = pca.transform(dif)
    pred = clf.predict_proba(np.concatenate((dif, add_features), axis=-1))[:,1]
    print(f"RF output shape {pred.shape}")
    gene_enhancers[info[6]] = []
    for j in range(len(pred)):
        if pred[j] > 0.9:
            gene_enhancers[info[6]].append(f"{mark_pos[j][0]}\t{mark_pos[j][0] - step // 2}\t{mark_pos[j][0] + step // 2}\t{pred[j]}\t{info[6]}")

    fig, axs = plt.subplots(1,1,figsize=(10, 4))
    x = range(len(pred))
    d1 = {'bin': x, 'expression': pred}
    df1 = pd.DataFrame(d1)
    sns.lineplot(data=df1, x='bin', y='expression', ax=axs)
    axs.fill_between(x, pred, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"enhancer_scape/{info[6]}.png")
    break



for gene in gene_enhancers.keys():
    with open(f"enhancer_list/{gene}", 'w+') as f:
        f.write('\n'.join(gene_enhancers[gene]))