import joblib
from main_params import MainParams
import model as mo
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import parse_data as parser
from scipy.ndimage.filters import gaussian_filter
import random


def recover_shape(v, size_X):
    v = np.asarray(v).flatten()
    end = int((size_X * size_X - size_X) / 2)
    v = v[:end]
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=2)] = v
    X = X + X.T
    return X


eval_gt_full = []
p = MainParams()
sp = "hg38"
heads = joblib.load(f"{p.pickle_folder}heads.gz")
hic_keys = p.hic_keys[sp]
hic_num = len(hic_keys)
for k in hic_keys:
    print(k, end=", ")

regions_tss = joblib.load(f"{p.pickle_folder}data_split.gz")[sp]["test"]
infos = random.sample(regions_tss, 100)

print(f"Number of positions: {len(infos)}")
one_hot = joblib.load(f"{p.pickle_folder}{sp}_one_hot.gz")

our_model, head_inds = mo.prepare_model(p, heads)

test_seq = []
hic_positions = []
for info in infos:
    pos_hic = info[1] - (info[1] % p.hic_bin_size)
    start = pos_hic - (pos_hic % p.bin_size) - p.half_size
    extra = start + p.input_size - len(one_hot[info[0]])
    if start < 0 or extra > 0:
        continue
    else:
        ns = one_hot[info[0]][start:start + p.input_size]
    hic_positions.append([info[0], pos_hic])
    test_seq.append(ns[:, :-1])

hic_output = parser.par_load_hic_data(hic_keys, p, hic_positions, 0)
hic_output = np.asarray(hic_output)
test_seq = np.asarray(test_seq, dtype=bool)
predictions_hic = []
for w in range(0, len(test_seq), p.w_step):
    print(w, end=" ")
    p1 = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
    predictions_hic.append(p1[head_inds[f"{sp}_hic"]])
predictions_hic = np.concatenate(predictions_hic)
print("drawing")
print(predictions_hic.shape)
print(hic_output.shape)
for n in range(len(hic_output)):
    fig, axs = plt.subplots(2, hic_num, figsize=(100, 10))
    for i in range(hic_num):
        mat = recover_shape(hic_output[n][i], p.num_hic_bins)
        mat = gaussian_filter(mat, sigma=1.0)
        sns.heatmap(mat, linewidth=0.0, ax=axs[0, i], square=True)
        axs[0, i].set(xticklabels=[])
        axs[0, i].set(yticklabels=[])

    for i in range(hic_num):
        mat = recover_shape(predictions_hic[n][i], p.num_hic_bins)
        sns.heatmap(mat, linewidth=0.0, ax=axs[1, i], square=True)
        axs[1, i].set(xticklabels=[])
        axs[1, i].set(yticklabels=[])

    fig.tight_layout()
    info = infos[n]
    plt.savefig(f"hic_check/{sp}_{info[0]}:{info[1] - p.half_size_hic}-{info[1] + p.half_size_hic}.png")
    plt.close(fig)