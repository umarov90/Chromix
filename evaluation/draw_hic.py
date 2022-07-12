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
head_name = "hg38"
heads = joblib.load(f"{p.pickle_folder}heads.gz")
head = heads[head_name]
hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0].tolist()
hic_num = len(hic_keys)
for k in hic_keys:
    print(k, end=", ")

infos = joblib.load(f"{p.pickle_folder}train_info.gz")[100:110]
# infos = infos[::20]
print(f"Number of positions: {len(infos)}")
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model =mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, head)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
    our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
    our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
    our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

hic_output = parser.par_load_hic_data(hic_keys, p, infos, 0)

test_seq = []
for info in infos:
    start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
    # if start in starts:
    #     continue
    # starts.append(start)
    extra = start + p.input_size - len(one_hot[info[0]])
    if start < 0:
        ns = one_hot[info[0]][0:start + p.input_size]
        ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
    elif extra > 0:
        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
        ns = np.concatenate((ns, np.zeros((extra, 5))))
    else:
        ns = one_hot[info[0]][start:start + p.input_size]
    if len(ns) != p.input_size:
        print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
    test_seq.append(ns[:, :-1])

test_seq = np.asarray(test_seq, dtype=bool)
for w in range(0, len(test_seq), p.w_step):
    print(w, end=" ")
    p1 = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
    if w == 0:
        predictions_hic = p1[3]
    else:
        predictions_hic = np.concatenate((predictions_hic, p1[3]))

hic_output = np.asarray(hic_output)
print("drawing")
print(predictions_hic.shape)
print(hic_output.shape)
hic_num = 5
for n in range(len(hic_output)):
    fig, axs = plt.subplots(2, hic_num, figsize=(20, 10))
    for i in range(hic_num):
        mat = recover_shape(hic_output[n][i], p.num_hic_bins)
        # mat = gaussian_filter(mat, sigma=0.5)
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
    plt.savefig(f"hic_check/{n}.svg")
    plt.close(fig)
