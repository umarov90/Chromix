import os 
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
import qnorm
from numba import jit
matplotlib.use("Agg")


p = MainParams()
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
head = joblib.load("pickle/heads.gz")["hg38"]
eval_track_names = []
for key in head.keys():
    eval_track_names += head[key]
def get_seq(info, input_size):
    start = info[1]
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

region_num_bins = 1024
test_info = pd.read_csv("data/human_test.bed", sep="\t")
test_info = test_info.sample(frac=0.04)
test_seq = []
load_info = []
for index, info in test_info.iterrows():
    seq = get_seq([info[0], info[1] + p.half_num_regions * p.bin_size - p.half_size], p.input_size)
    load_info.append([info[0], info[1] // p.bin_size, info[1] // p.bin_size + region_num_bins])
    test_seq.append(seq)

test_seq = np.asarray(test_seq, dtype=bool)
gt = parser.par_load_data(load_info, eval_track_names, p)
print(gt.shape)
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import model as mo
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 6, p.hic_size, head)
    our_model.get_layer("our_stem").set_weights(joblib.load(p.model_path + "_stem"))
    our_model.get_layer("our_body").set_weights(joblib.load(p.model_path + "_body"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
    our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
    our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
    our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

for w in range(0, len(test_seq), p.w_step):
    print(w, end=" ")
    pr = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
    pr = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
    pr = pr[:, :, :region_num_bins]
    if w == 0:
        predictions = pr
    else:
        predictions = np.concatenate((predictions, pr), dtype=np.float16)

# predictions = np.log10(predictions + 1)
print(predictions.shape)
meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
track_types = {}
for track in eval_track_names:
    meta_row = meta.loc[meta['file_name'] == track]
    track_types[track] = meta_row.iloc[0]["technology"]

# Across genes
corrs_p = {}
corrs_s = {}
all_track_spearman = {}
for i, track in enumerate(eval_track_names):
    if i % 500 == 0:
        print(i, end=" ")
    a = []
    b = []
    for j in range(len(test_info)):
        a.append(predictions[j,i].flatten())
        b.append(gt[j,i].flatten())
    a = np.nan_to_num(a, neginf=0, posinf=0)
    b = np.nan_to_num(b, neginf=0, posinf=0)            
    pc = stats.pearsonr(a.flatten(), b.flatten())[0]
    sc = stats.spearmanr(a.flatten(), b.flatten())[0]
    if not math.isnan(sc) and not math.isnan(pc):
        corrs_p.setdefault(track_types[track], []).append((pc, track))
        corrs_s.setdefault(track_types[track], []).append((sc, track))
        all_track_spearman[track] = stats.spearmanr(a.flatten(), b.flatten())[0]

print("")
print("Type\tCount\tAcross genes PCC\tAcross genes SC")
for track_type in corrs_p.keys():
    type_pcc = [i[0] for i in corrs_p[track_type]]
    print(f"{track_type}\t{len(type_pcc)}\t{np.mean(type_pcc):.2f}\t"
          f"{np.mean([i[0] for i in corrs_s[track_type]]):.2f}\t")