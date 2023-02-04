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
import qnorm
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
# p.NUM_GPU = 1
# p.GLOBAL_BATCH_SIZE = p.NUM_GPU * p.BATCH_SIZE
# p.predict_batch_size = 2 * p.GLOBAL_BATCH_SIZE

model_path = 'https://tfhub.dev/deepmind/enformer/1'
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
SEQUENCE_LENGTH = 393216

train_info_all, valid_info_all, test_info_all, _ = parser.parse_sequences(p)
all_info = train_info_all + valid_info_all + test_info_all
test_info = []
for info in test_info_all: # test_info_all
    if info[5]: 
        continue
    # if info[0] != "chr14":
    #     continue
    test_info.append(info)
    # if len(test_info) > 6000:
    #     break
cor_tracks = pd.read_csv("data/fantom_tracks.tsv", sep="\t").iloc[:, 0].tolist()
# test_info = test_info[:100]
enf_tracks = df_targets[df_targets['description'].str.contains("CAGE")]['identifier'].tolist()
heads = joblib.load("pickle/heads.gz")
head_tracks = heads["expression"]

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
pred_matrix = joblib.load("pred_matrix.p")
# class Enformer:

#     def __init__(self, tfhub_url):
#         self._model = hub.load(tfhub_url).model

#     def predict_on_batch(self, inputs):
#         predictions = self._model.predict_on_batch(inputs)
#         return {k: v.numpy() for k, v in predictions.items()}

#     @tf.function
#     def contribution_input_grad(self, input_sequence,
#                                 target_mask, output_head='human'):
#         input_sequence = input_sequence[tf.newaxis]

#         target_mask_mass = tf.reduce_sum(target_mask)
#         with tf.GradientTape() as tape:
#             tape.watch(input_sequence)
#             prediction = tf.reduce_sum(
#                 target_mask[tf.newaxis] *
#                 self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

#         input_grad = tape.gradient(prediction, input_sequence) * input_sequence
#         input_grad = tf.squeeze(input_grad, axis=0)
#         return tf.reduce_sum(input_grad, axis=-1)
# model = Enformer(model_path)
# pred_matrix = np.zeros((len(eval_tracks), len(test_info)))
# # counts = [0, 0, 0]
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
#         pred_matrix[j, index] = gene_expression

# print("")
# # print(counts)
# end = time.time()
# print("Enformer time")
# print(end - start)
# joblib.dump(pred_matrix, "pred_matrix.p", compress=3)
qnorm_axis = 0
# pred_matrix = qnorm.quantile_normalize(pred_matrix, axis=qnorm_axis)
# OUR MODEL #####################################################################################################
#################################################################################################################
# pred_matrix_our = joblib.load("pred_matrix_our.p")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
heads = joblib.load(f"{p.pickle_folder}heads.gz")
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads["expression"])
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
pred_matrix_our = np.zeros((len(eval_tracks), len(test_info)))

test_seq = []
for index, info in enumerate(test_info):
    seq = get_seq([info[0], info[1]], p.input_size)
    test_seq.append(seq)
test_seq = np.asarray(test_seq, dtype=bool)
start = time.time()
for w in range(0, len(test_seq), p.w_step):
    print(w, end=" ")
    pr = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
    p2 = pr[:, :, p.mid_bin]
    if w == 0:
        predictions = p2
    else:
        predictions = np.concatenate((predictions, p2), dtype=np.float16)

for index, info in enumerate(test_info):
    for j, track in enumerate(eval_tracks):
        t = track_ind_our[track]
        pred_matrix_our[j, index] = predictions[index, t]
print("")
end = time.time()
print("Our time")
print(end - start)
joblib.dump(pred_matrix_our, "pred_matrix_our.p", compress=3)
print(f"{np.max(pred_matrix_our)}\t{np.std(pred_matrix_our)}\t{np.mean(pred_matrix_our)}\t{np.median(pred_matrix_our)}")
# pred_matrix_our = qnorm.quantile_normalize(pred_matrix_our, axis=qnorm_axis)

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
load_info = []
for j, info in enumerate(all_info):
    load_info.append([info[0], info[1] // p.bin_size])
gt_matrix_all = parser.par_load_data(load_info, eval_track_names, p).T
gt_matrix_all = gt_matrix_all.astype(np.float32)
gt_matrix_all = np.sum(gt_matrix_all, axis=1)
print(gt_matrix_all.shape)

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
            # a.append((final_pred[j, i] * 1000000) / gt_matrix_all[j])
            # b.append((eval_gt[j, i] * 1000000) / gt_matrix_all[j])
            a.append(final_pred[j, i])
            b.append(eval_gt[j, i])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.mean(b) < 10 or np.var(b) < 10:
            continue
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
    
np.savetxt("gt_matrix.csv", np.mean(gt_matrix, axis=0), delimiter=",")
np.savetxt("pred_matrix_enformer.csv", np.mean(pred_matrix, axis=0), delimiter=",")
np.savetxt("pred_matrix_our.csv", np.mean(pred_matrix_our, axis=0), delimiter=",")
print("Enformer =================================================")
scatter_data_enf = eval_perf(gt_matrix, pred_matrix)
print("OUR =================================================")
scatter_data_our = eval_perf(gt_matrix, pred_matrix_our)


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

# gt = np.zeros((len(eval_tracks), len(eval_gt.keys())))
# enf = np.zeros((len(eval_tracks), len(eval_gt.keys())))
# chromix = np.zeros((len(eval_tracks), len(eval_gt.keys())))

# for i, track in enumerate(eval_tracks):
#     for j, gene in enumerate(eval_gt.keys()):
#         gt[i][j] = eval_gt[gene][track]
#         enf[i][j] = final_pred_enformer[gene][track]
#         chromix[i][j] = final_pred_our[gene][track]

# gt = qnorm.quantile_normalize(gt, axis=0)
# # gt = gt - gt.mean(axis=-1, keepdims=True)
# enf = qnorm.quantile_normalize(enf, axis=0)
# # enf = enf - enf.mean(axis=-1, keepdims=True)
# chromix = qnorm.quantile_normalize(chromix, axis=0)
# # chromix = chromix - chromix.mean(axis=-1, keepdims=True)



# print("Enformer =================================================")
# eval_perf_norm(gt, enf)
# print("OUR =================================================")
# eval_perf_norm(gt, chromix)

# def plot_umap(mat, ax, title):
#     reducer = umap.UMAP()
#     umap1 = reducer.fit_transform(mat)
#     data = {'x': umap1[:, 0],
#             'y': umap1[:, 1]}

#     df = pd.DataFrame(data)

#     sns.scatterplot(x="x", y="y", data=df, s=5, alpha=0.2, ax=ax)
#     for i, track in enumerate(cor_tracks):
#         if track in track_ind_eval.keys():
#             ax.text(umap1[track_ind_eval[track], 0], umap1[track_ind_eval[track], 1], short_name[track], color="black", fontsize=6)
#     ax.set_title(title)
#     ax.set_xlabel("A1")
#     ax.set_ylabel("A2")

# fig, axs = plt.subplots(1,3,figsize=(15, 5))
# plot_umap(gt_matrix, axs[0], "GT")
# plot_umap(pred_matrix, axs[1], "Enformer")
# plot_umap(pred_matrix_our, axs[2], "Our")
# plt.tight_layout()
# plt.savefig("umaps.svg")