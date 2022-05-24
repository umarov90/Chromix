import gc
import os
import pathlib
import tensorflow as tf
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
from mpl_toolkits.axisartist.grid_finder import DictFormatter
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib import colors
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


def recover_shape(v, size_X):
    v = np.asarray(v).flatten()
    end = int((size_X * size_X - size_X) / 2)
    v = v[:end]
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=1)] = v
    X = X + X.T
    return X


eval_gt_full = []
p = MainParams()
w_step = 20
predict_batch_size = 4
script_folder = pathlib.Path(__file__).parent.resolve()
heads = joblib.load(f"{p.pickle_folder}/heads.gz")
head_tracks = heads["hg38"]
hic_keys = parser.parse_hic(p)
for k in hic_keys:
    print(k, end=", ")
# hn = []
# for h in [3, 7, 10, 15]:
#     print(hic_keys[h])
#     hn.append(hic_keys[h])
# hn.append("hic_HeLa.10kb.intra_chromosomal.interaction_table.tsv")
# hic_keys = hn
# joblib.dump(hic_keys, "pickle/hic_keys.gz", compress=3)

infos = joblib.load("pickle/test_info.gz")[100:600]
infos = infos[::40]
print(f"Number of positions: {len(infos)}")

hic_output = []
hic_output2 = []
for hi, key in enumerate(hic_keys):
    hdf = joblib.load(p.parsed_hic_folder + key)
    ni = 0
    for i, info in enumerate(infos):
        hd = hdf[info[0]]
        hic_mat = np.zeros((p.num_hic_bins, p.num_hic_bins))
        start_hic = int(info[1] - (info[1] % p.bin_size) - p.half_size_hic)
        end_hic = start_hic + 2 * p.half_size_hic
        start_row = hd['start1'].searchsorted(start_hic - p.hic_bin_size, side='left')
        end_row = hd['start1'].searchsorted(end_hic, side='right')
        hd = hd.iloc[start_row:end_row]
        # convert start of the input region to the bin number
        start_hic = int(start_hic / p.hic_bin_size)
        # subtract start bin from the binned entries in the range [start_row : end_row]
        l1 = (np.floor(hd["start1"].values / p.hic_bin_size) - start_hic).astype(int)
        l2 = (np.floor(hd["start2"].values / p.hic_bin_size) - start_hic).astype(int)
        hic_score = hd["score"].values
        # drop contacts with regions outside the [start_row : end_row] range
        lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
        l1 = l1[lix]
        l2 = l2[lix]
        hic_score = hic_score[lix]
        hic_mat[l1, l2] += hic_score

        hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
        hic_mat = gaussian_filter(hic_mat, sigma=1)
        hic_mat2 = np.rot90(hic_mat, k=2)
        hic_mat2 = hic_mat2[np.triu_indices_from(hic_mat2, k=1)]

        hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
        if hi == 0:
            hic_output.append([])
            hic_output2.append([])
        hic_output[ni].append(hic_mat)
        hic_output2[ni].append(hic_mat2)
        ni += 1
    del hd
    del hdf
    gc.collect()


for n in range(len(hic_output)):
    fig, axs = plt.subplots(2, len(hic_keys), figsize=(60, 20))
    for i in range(len(hic_keys)):
        mat = recover_shape(hic_output[n][i], p.num_hic_bins)
        sns.heatmap(mat, linewidth=0.0, ax=axs[0, i], square=True, cbar=False)
        axs[0, i].set(xticklabels=[])
        axs[0, i].set(yticklabels=[])
        axs[0, i].set_title("hic"+str(i+1))

        mat = recover_shape(hic_output2[n][i], p.num_hic_bins)
        sns.heatmap(mat, linewidth=0.0, ax=axs[1, i], square=True, cbar=False)
        axs[1, i].set(xticklabels=[])
        axs[1, i].set(yticklabels=[])
        axs[1, i].set_title("hic" + str(i + 1))

    fig.tight_layout()
    plt.savefig(f"hic_check/{n}.png")
    plt.close(fig)


