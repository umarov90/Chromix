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
    X[np.triu_indices(X.shape[0], k=2)] = v
    X = X + X.T
    return X


eval_gt_full = []
p = MainParams()
head = joblib.load(f"{p.pickle_folder}heads.gz")["hg38"]
hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0].tolist()
hic_num = len(hic_keys)
for k in hic_keys:
    print(k, end=", ")

regions_tss = joblib.load(f"{p.pickle_folder}test_info.gz")
# regions_tss = joblib.load(f"{p.pickle_folder}train_info.gz")
infos = random.sample(regions_tss, 100)

print(f"Number of positions: {len(infos)}")
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, head)
    our_model.get_layer("our_stem").set_weights(joblib.load(p.model_path + "_stem"))
    our_model.get_layer("our_body").set_weights(joblib.load(p.model_path + "_body"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
    our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
    our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
    our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

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
    predictions_hic.append(p1[-1])
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
        # mat = gaussian_filter(mat, sigma=0.5)
        sns.heatmap(mat, linewidth=0.0, ax=axs[1, i], square=True)
        axs[1, i].set(xticklabels=[])
        axs[1, i].set(yticklabels=[])

    fig.tight_layout()
    info = infos[n]
    plt.savefig(f"hic_check/{info[0]}:{info[1] - 500000}-{info[1] + 500000}.png")
    plt.close(fig)
