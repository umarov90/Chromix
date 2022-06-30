import gc
import os
import pathlib
import joblib
from main_params import MainParams
import visualization as viz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import parse_data as parser
from scipy.ndimage.filters import gaussian_filter
import cooler
import pandas as pd
import shutil
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
# matplotlib.use('Qt5Agg')


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
w_step = 20
predict_batch_size = 4
script_folder = pathlib.Path(__file__).parent.resolve()
heads = joblib.load(f"{p.pickle_folder}/heads.gz")
head_tracks = heads["hg38"]

one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
train_info = joblib.load("pickle/train_info.gz")
train_eval_chr = "chr1"
infos = []
for info in train_info:
    if info[0] == train_eval_chr:
        infos.append(info)
infos = infos[::400]
print(f"Number of positions: {len(infos)}")

hic_output = []
hic_output2 = []
n_bins = 50
nan_counts = {}
good_hic = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0].tolist()
directory = "hic"

for i, info in enumerate(infos):
    hic_output.append([])
    hic_output2.append([])

for filename in good_hic: # os.listdir(directory)
    if not filename.endswith(".mcool"):
        continue
    c = cooler.Cooler("hic/" + filename + "::resolutions/5000")
    resolution = c.binsize
    print(filename)
    for i, info in enumerate(infos):
        start_hic = int(info[1] - (info[1] % p.bin_size) - p.half_size_hic) + p.hic_bin_size // 2
        end_hic = start_hic + 2 * p.half_size_hic - p.hic_bin_size
        dif = end_hic - start_hic

        if start_hic < 0 or end_hic >= len(one_hot[info[0]]):
            continue
        hic_mat = c.matrix(balance=True, field="count").fetch(f'{info[0]}:{start_hic}-{end_hic}')
        # count = np.count_nonzero(np.isnan(hic_mat))
        # nan_counts[filename] = nan_counts[filename] + count if filename in nan_counts else count

        hic_mat[np.isnan(hic_mat)] = 0

        hic_mat = hic_mat - np.diag(np.diag(hic_mat, k=1), k=1) - np.diag(np.diag(hic_mat, k=-1), k=-1) - np.diag(np.diag(hic_mat))

        # sns.heatmap(hic_mat, linewidth=0.0, square=True)
        # plt.show()

        hic_mat = gaussian_filter(hic_mat, sigma=1)
        hic_mat2 = np.rot90(hic_mat, k=2)

        hic_mat2 = hic_mat2[np.triu_indices_from(hic_mat2, k=2)]
        hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=2)]

        hic_output[i].append(hic_mat)
        hic_output2[i].append(hic_mat2)
print("Drawing")
# for key in nan_counts:
#     print(f"{key}: {nan_counts[key]}")
#
# sorted_dict = {k: v for k, v in sorted(nan_counts.items(), key=lambda item: item[1])}
# with open("good_hic.tsv", 'w+') as f:
#     f.write('\n'.join(list(sorted_dict.keys())[:int(0.1 * len(sorted_dict.keys()))]))
# exit()
sns.set(font_scale=0.5)
for n in range(len(hic_output)):
    fig, axs = plt.subplots(1, len(good_hic), figsize=(100, 10))
    for i in range(len(good_hic)):
        mat = recover_shape(hic_output[n][i], n_bins)
        sns.heatmap(mat, linewidth=0.0, ax=axs[i], square=True, cbar=False)
        axs[i].set(xticklabels=[])
        axs[i].set(yticklabels=[])
        axs[i].set_title(good_hic[i])

        # mat = recover_shape(hic_output2[n][i], n_bins)
        # sns.heatmap(mat, linewidth=0.0, ax=axs[1, i], square=True, cbar=False)
        # axs[1, i].set(xticklabels=[])
        # axs[1, i].set(yticklabels=[])
        # axs[1, i].set_title("hic" + str(i + 1))

    fig.tight_layout()
    plt.savefig(f"hic_check/{n}.png")
    plt.close(fig)
    print(n)


