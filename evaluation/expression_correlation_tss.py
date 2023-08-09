import math
import random

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
import seaborn as sns
from pathlib import Path
import qnorm
import torch
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


def get_seq(info, input_size, add_half=False):
    start = int(info[1] - (info[1] % p.bin_size) - input_size // 2)
    if add_half:
        start = start + p.bin_size // 2
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


p = MainParams()
heads = joblib.load("pickle/heads.gz")
head_tracks = heads["hg38"]["expression"]
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
test_info = joblib.load(f"{p.pickle_folder}data_split.gz")["hg38"]["test"]
if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
    track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
else:
    track_names_col = parser.parse_tracks(p)


df_targets = pd.read_csv("data/targets_human.txt", sep='\t')
SEQUENCE_LENGTH = 393216

# test_info = random.sample(test_info, 100)
enf_tracks = df_targets[df_targets['description'].str.contains("CAGE")]['identifier'].tolist()

short_name = {}
for i, t in enumerate(head_tracks):
    track_type = t[:t.index(".")]
    start = t.index(".ctss.") + len(".ctss.")
    end = t.find("_", t.find("_") + 1) # t.find("_")
    if end == -1:
        end = t.find(".", start)
    short_name[t] = t[start:end]

track_ind_our = {}
track_ind_eval = {}
full_name = {}
eval_tracks = []
for track in enf_tracks:
    for our_track in head_tracks:
        if track in our_track:
            track_ind_our[track] = head_tracks.index(our_track)
            eval_tracks.append(track)
            track_ind_eval[our_track] = eval_tracks.index(track)
            full_name[track] = our_track
            break


print(f"Eval tracks {len(eval_tracks)}")
print(f"TSS {len(test_info)}")

track_ind = {}
for j, track in enumerate(eval_tracks):
    track_ind[track] = df_targets[df_targets['identifier'].str.contains(track)].index

# ENFORMER #####################################################################################################
################################################################################################################
pred_matrix_enformer = joblib.load("pred_matrix_enformer.p")
# class Enformer:
#
#     def __init__(self):
#         # self._model = hub.load('https://tfhub.dev/deepmind/enformer/1').model
#         # tf.saved_model.save(self._model, "data/enformer_model")
#         self._model = tf.saved_model.load("data/enformer_model")
#
#     def predict_on_batch(self, inputs):
#         predictions = self._model.predict_on_batch(inputs)
#         return {k: v.numpy() for k, v in predictions.items()}
#
# model = Enformer()
#
# pred_matrix_enformer = np.zeros((len(eval_tracks), len(test_info)))
# counts = [0, 0, 0]
# print("Predicting")
# start = time.time()
# for index, info in enumerate(test_info):
#     if index % 100 == 0:
#         print(index, end=" ")
#     # batch = []
#     # for rvc in [True, False]:
#     # for i in range(-1, 2, 1):
#     sequence_one_hot = get_seq(info, SEQUENCE_LENGTH)
#     # if rvc:
#     #     sequence_one_hot = rev_comp(sequence_one_hot)
#     # batch.append(sequence_one_hot)
#     prediction = np.mean(model.predict_on_batch(sequence_one_hot[np.newaxis])['human'], axis=0)
#     for j, track in enumerate(eval_tracks):
#         t = track_ind[track]
#         bins = prediction[448, t] # [prediction[447, t], prediction[448, t], prediction[449, t]]
#         gene_expression = np.sum(bins)
#         # counts[bins.index(max(bins))] += 1
#         pred_matrix_enformer[j, index] = gene_expression
#
# print("")
# # print(counts)
# end = time.time()
# print("Enformer time")
# print(end - start)
# joblib.dump(pred_matrix_enformer, "pred_matrix_enformer.p", compress=3)
# qnorm_axis = 0
# pred_matrix_enformer = qnorm.quantile_normalize(pred_matrix_enformer, axis=qnorm_axis)
# OUR MODEL #####################################################################################################
#################################################################################################################
pred_matrix_chromix = joblib.load("pred_matrix_chromix.p")
# pred_matrix_chromix = np.zeros((len(eval_tracks), len(test_info)))
# test_seq = []
# for index, info in enumerate(test_info):
#     seq = get_seq([info[0], info[1]], p.input_size)
#     test_seq.append(seq)
# test_seq = np.asarray(test_seq, dtype=bool)
# start = time.time()
#
# model, head_inds = mo.prepare_model(p, heads)
#
# predictions = []
# for w in range(0, len(test_seq), p.w_step):
#     print(w, end=" ")
#     pr = model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
#     p2 = pr[head_inds["hg38_expression"]][:, :, p.mid_bin]
#     predictions.append(p2)
# predictions = np.concatenate(predictions)
# for index, info in enumerate(test_info):
#     for j, track in enumerate(eval_tracks):
#         t = track_ind_our[track]
#         pred_matrix_chromix[j, index] = predictions[index, t]
# print("")
# end = time.time()
# print("Our time")
# print(end - start)
# joblib.dump(pred_matrix_chromix, "pred_matrix_chromix.p", compress=3)
# pred_matrix_chromix = qnorm.quantile_normalize(pred_matrix_chromix, axis=qnorm_axis)
# print(f"{np.max(pred_matrix_chromix)}\t{np.std(pred_matrix_chromix)}\t{np.mean(pred_matrix_chromix)}\t{np.median(pred_matrix_chromix)}")

# GT DATA #####################################################################################################
#################################################################################################################
eval_track_names = []
for track in eval_tracks:
    eval_track_names.append(full_name[track])
load_info = []
for j, info in enumerate(test_info):
    load_info.append([info[0], info[1] // p.bin_size])
print("Loading ground truth tracks")
gt_matrix = parser.par_load_data(load_info, eval_track_names, p).T
print(gt_matrix.shape)

# gt_matrix = qnorm.quantile_normalize(gt_matrix, axis=qnorm_axis)

print("")
def eval_perf(eval_gt, final_pred):
    scatter_data = []
    corr_s = []
    corr_p = []
    for i in range(final_pred.shape[0]):
        a = []
        b = []
        for j in range(final_pred.shape[1]):
            # if eval_gt[i, j] < 384:
            a.append(final_pred[i, j])
            b.append(eval_gt[i, j])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.sum(b)==0:
            continue
        sc = stats.spearmanr(a, b)[0]
        pc = stats.pearsonr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
            corr_p.append(pc)
        else:
            corr_s.append(0)
            corr_p.append(0)

    print("")
    print(f"Across genes {len(corr_s)} {np.mean(corr_s)} {np.mean(corr_p)}")
    scatter_data.append(np.asarray(corr_s))
    corr_s = []
    corr_p = []
    for i in range(final_pred.shape[1]):
        a = []
        b = []
        for j in range(final_pred.shape[0]):
            a.append(final_pred[j, i])
            b.append(eval_gt[j, i])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        # if np.mean(b) < 1 or np.std(b) < 0.1:
        #     continue
        if np.sum(b)==0:
            continue
        sc = stats.spearmanr(a, b)[0]
        pc = stats.pearsonr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
            corr_p.append(pc)
        else:
            corr_s.append(0)
            corr_p.append(0)
    print(f"Across tracks {len(corr_s)} {np.mean(corr_s)} {np.mean(corr_p)}")
    scatter_data.append(np.asarray(corr_s))
    return scatter_data


print("Enformer =================================================")
scatter_data_enf = eval_perf(gt_matrix, pred_matrix_enformer)
print("OUR =================================================")
scatter_data_our = eval_perf(gt_matrix, pred_matrix_chromix)


fig, axs = plt.subplots(1,2,figsize=(20, 10))
r, p = stats.pearsonr(scatter_data_our[0], scatter_data_enf[0])
sns.scatterplot(x=scatter_data_our[0], y=scatter_data_enf[0], ax=axs[0], label="r = {0:.2f}; p = {1:.2e}".format(r, p))
axs[0].axline((0, 0), (1, 1), color='r', lw=2)
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_title("Across tracks")
axs[0].set(xlabel='Chromix', ylabel='Enformer')
r, p = stats.pearsonr(scatter_data_our[1], scatter_data_enf[1])
sns.scatterplot(x=scatter_data_our[1], y=scatter_data_enf[1], ax=axs[1], label="r = {0:.2f}; p = {1:.2e}".format(r, p))
axs[1].axline((0, 0), (1, 1), color='r', lw=2)
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].set_title("Across genes")
axs[1].set(xlabel='Chromix', ylabel='Enformer')
plt.suptitle('Enformer comparision')
plt.tight_layout()
plt.savefig("predictions_scatter.svg")


# sns.set(font_scale = 2.5)
# t, p = ttest_ind(scatter_data_our[1], scatter_data_enf[1])
# print(f"Chromix/Enformer ttest: {t} {p}")

# fig, axs = plt.subplots(1,1,figsize=(10, 10))

# df1 = pd.DataFrame(np.asarray([scatter_data_our[1], ["Chromix"] * len(scatter_data_our[1])]).T, columns=['Correlation', 'Method'])
# df2 = pd.DataFrame(np.asarray([scatter_data_enf[1], ["Enformer"] * len(scatter_data_enf[1])]).T, columns=['Correlation', 'Method'])

# merged = pd.concat([df1, df2], ignore_index=True, axis=0)
# merged.reset_index(drop=True, inplace=True)
# merged['Correlation'] = merged['Correlation'].astype(float)
# print(merged.shape)
# sns.histplot(data=merged, x="Correlation", hue="Method", ax=axs, bins=50, element="step", binrange=(0.75, 1.0))
# axs.text(0.76, 90, "t = {0:.2f}\np = {1:.2e}".format(t, p), horizontalalignment='left', size='medium', color='black', weight='semibold')
# plt.title("Gene expression prediction")
# plt.tight_layout()
# plt.savefig("predictions_histplot.svg")

# lib_dict = pd.read_csv('data/lib_size.csv', sep=",", header=None, index_col=0, squeeze=True).to_dict()

# sizes = []
# for track in eval_tracks:
#     sizes.append(int(lib_dict[full_name[track]]))


# fig, axs = plt.subplots(1,2,figsize=(10, 5))
# r, p = stats.pearsonr(scatter_data_our[1], sizes)
# sns.scatterplot(x=scatter_data_our[1], y=sizes, ax=axs[0], label="r = {0:.2f}; p = {1:.2e}".format(r, p))
# axs[0].set_title("Chromix")
# axs[0].set(xlabel='Chromix', ylabel='Library size')
# r, p = stats.pearsonr(scatter_data_enf[1], sizes)
# sns.scatterplot(x=scatter_data_enf[1], y=sizes, ax=axs[1], label="r = {0:.2f}; p = {1:.2e}".format(r, p))
# axs[1].set_title("Enformer")
# axs[1].set(xlabel='Enformer', ylabel='Library size')
# plt.suptitle('Library size analysis')
# plt.tight_layout()
# plt.savefig("lib_size_scatter.svg")
# plt.clf()