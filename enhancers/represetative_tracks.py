import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
import multiprocessing as mp
import math
import tensorflow_hub as hub
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
import model as mo
from main_params import MainParams
import time
import parse_data as parser
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
import umap
from sklearn.decomposition import PCA
import sklearn
import qnorm
import leidenalg
import igraph
matplotlib.use("Agg")


def rev_comp(s):
    reversed_arr = s[::-1]
    vals = []
    for v in reversed_arr:
        if v[0]:
            vals.append([0, 0, 0, 1])
        elif v[1]:
            vals.append([0, 0, 1, 0])
        elif v[2]:
            vals.append([0, 1, 0, 0])
        elif v[3]:
            vals.append([1, 0, 0, 0])
        else:
            vals.append([0, 0, 0, 0])
    return np.array(vals, dtype=np.float32)


p = MainParams()
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
def get_seq(info, input_size, sub_half=False):
    start = int(info[1] - (info[1] % p.bin_size) - input_size // 2)
    if sub_half:
        start = start - p.bin_size // 2
    extra = start + input_size - len(one_hot[info[0]])
    if start < 0:
        ns = one_hot[info[0]][0:start + input_size]
        ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
    elif extra > 0:
        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
        ns = np.concatenate((ns, np.zeros((extra, 5))))
    else:
        ns = one_hot[info[0]][start:start + input_size]
    return ns[:, :-1]

_, _, test_info, _ = parser.parse_sequences(p)

head_name = "hg38"
heads = joblib.load(f"{p.pickle_folder}heads.gz")

pred_matrix_our = joblib.load("pred_matrix_our.p")
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
# heads = joblib.load(f"{p.pickle_folder}heads.gz")
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads["hg38"])
#     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
#     # our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
#     # our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))

# test_seq = []
# for index, info in enumerate(test_info):
#     seq = get_seq([info[0], info[1]], p.input_size)
#     test_seq.append(seq)
# test_seq = np.asarray(test_seq, dtype=bool)
# start = time.time()
# for w in range(0, len(test_seq), p.w_step):
#     print(w, end=" ")
#     pr = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
#     p2 = pr[0][:, :, p.mid_bin - 4 : p.mid_bin + 4]
#     p2 = np.sum(p2, axis=-1)
#     if w == 0:
#         predictions = p2
#     else:
#         predictions = np.concatenate((predictions, p2), dtype=np.float16)

# pred_matrix_our = predictions.T
# print("")
# end = time.time()
# print("Our time")
# print(end - start)
# joblib.dump(pred_matrix_our, "pred_matrix_our.p", compress=3)

# GT DATA #####################################################################################################
#################################################################################################################
load_info = []
for j, info in enumerate(test_info):
    load_info.append([info[0], info[1] // p.bin_size - 4, info[1] // p.bin_size + 4])
print("Loading ground truth tracks")
gt_matrix = np.sum(parser.par_load_data(load_info, heads["hg38"]["expression"], p), axis=-1).T
print(gt_matrix.shape)

def eval_perf(eval_gt, final_pred):
    corr_s = []
    for i in range(final_pred.shape[0]):
        a = []
        b = []
        for j in range(final_pred.shape[1]):
            a.append(final_pred[i, j])
            b.append(eval_gt[i, j])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.sum(b)==0:
            continue
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
        else:
            corr_s.append(0)
    return corr_s
    

track_correlation = eval_perf(gt_matrix, pred_matrix_our)

A = sklearn.neighbors.kneighbors_graph(pred_matrix_our, n_neighbors=10, mode='distance', n_jobs=10).toarray() 

# Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
g = igraph.Graph.Adjacency((A > 0).tolist())

# Add edge weights and node labels.
g.es['weight'] = A[A.nonzero()]
g.vs['label'] = heads["hg38"]["expression"]

part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
joblib.dump(part, "leiden_part.p", compress=3)
