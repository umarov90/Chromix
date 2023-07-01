import gc
import os
import pathlib
import joblib
from main_params import MainParams
import visualization as viz
import model as mo
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import parse_data as parser
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from scipy import stats
import random
import torch
from torch.utils.data import DataLoader
from torch import autocast, nn, optim
from sync_batchnorm import convert_model


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
heads = joblib.load(f"{p.pickle_folder}heads.gz")
for head_key in heads.keys():
    if isinstance(heads[head_key], dict):
        for key2 in heads[head_key].keys():
            print(f"Number of tracks in head {head_key} {key2}: {len(heads[head_key][key2])}")
            p.output_heads[head_key + "_" + key2] = len(heads[head_key][key2])
    else:
        print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")
        p.output_heads[head_key + "_expression"] = len(heads[head_key])

hic_keys = p.hic_keys["hg38"]
hic_num = len(hic_keys)
for k in hic_keys:
    print(k, end=", ")

regions_tss = joblib.load(f"{p.pickle_folder}test_info.gz")
# regions_tss = joblib.load(f"{p.pickle_folder}train_info.gz")
infos = random.sample(regions_tss, 100)

print(f"Number of positions: {len(infos)}")
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
model = mo.Chromix(p)
model = nn.DataParallel(model)
model = convert_model(model).to("cuda:0")
mo.load_weights(p, model)

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

hic_output = parser.par_load_hic_data(hic_keys, p, hic_positions, 0, swap=False)
hic_output = np.asarray(hic_output)
test_seq = np.asarray(test_seq, dtype=bool)
dd = mo.DatasetDNA(test_seq)
ddl = DataLoader(dataset=dd, batch_size=p.pred_batch_size, shuffle=False)
predictions_hic = []
model.eval()
for batch, X in enumerate(ddl):
    print(batch, end=" ")
    with torch.no_grad():
        pr = model(X)
    predictions_hic.append(pr['hg38_hic'].cpu().numpy())
predictions_hic = np.concatenate(predictions_hic)
predictions_hic = predictions_hic.swapaxes(1, 2)
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
        # mat = gaussian_filter(mat, sigma=0.5)
        sns.heatmap(mat, linewidth=0.0, ax=axs[1, i], square=True)
        axs[1, i].set(xticklabels=[])
        axs[1, i].set(yticklabels=[])

    fig.tight_layout()
    info = infos[n]
    plt.savefig(f"hic_check/{info[0]}:{info[1] - 100000}-{info[1] + 100000}.png")
    plt.close(fig)
