import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
import multiprocessing as mp
import math
import tensorflow_hub as hub
import joblib
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
import pyBigWig
import model as mo
from main_params import MainParams
import time
import parse_data as parser
import matplotlib
import matplotlib.pyplot as plt
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
# p.NUM_GPU = 1
# p.GLOBAL_BATCH_SIZE = p.NUM_GPU * p.BATCH_SIZE
# p.predict_batch_size = 2 * p.GLOBAL_BATCH_SIZE

model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = f'{p.data_folder}data/hg38/genome.fa'
clinvar_vcf = '/root/data/clinvar.vcf.gz'

# Download targets from Basenji2 dataset
# Cite: Kelley et al Cross-species regulatory sequence activity prediction. PLoS Comput. Biol. 16, e1008050 (2020).
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')

pyfaidx.Faidx(fasta_file)

SEQUENCE_LENGTH = 393216


class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


model = Enformer(model_path)

fasta_extractor = FastaStringExtractor(fasta_file)

_, _, test_info_all, _ = parser.parse_sequences(p)
test_info = []
for info in test_info_all:
    if info[5]:
        continue
    # if info[0] != "chr14":
    #     continue
    test_info.append(info)
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
#################################################################################################################

pred_matrix = joblib.load("pred_matrix.p")
# print(f"{np.max(pred_matrix)}\t{np.std(pred_matrix)}\t{np.mean(pred_matrix)}\t{np.median(pred_matrix)}")
# pred_matrix = np.zeros((len(eval_tracks), len(test_info)))
# counts = [0, 0, 0, 0, 0]
# print("Predicting")
# start = time.time()
# for index, info in enumerate(test_info):
#     if index % 100 == 0:
#         print(index, end=" ")
#     batch = []
#     # for rvc in [True, False]:
#     # for i in range(-1, 2, 1):
#     # target_interval = kipoiseq.Interval(info[0], info[1] - 63 + i, info[1] - 63 + i + 1)
#     target_interval = kipoiseq.Interval(info[0], info[1] - 63, info[1] - 63 + 1)
#     sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
#     # if rvc:
#     #     sequence_one_hot = rev_comp(sequence_one_hot)
#     batch.append(sequence_one_hot)
#     prediction = np.mean(model.predict_on_batch(batch)['human'], axis=0)
#     for j, track in enumerate(eval_tracks):
#         t = track_ind[track]
#         bins = [prediction[447, t], prediction[448, t], prediction[449, t]]
#         gene_expression = np.sum(bins)
#         counts[bins.index(max(bins))] += 1
#         pred_matrix[j, index] = gene_expression

# print("")
# print(counts)
# end = time.time()
# print("Enformer time")
# print(end - start)
# joblib.dump(pred_matrix, "pred_matrix.p", compress=3)
final_pred_enformer = {}
for i in range(len(test_info)):
    final_pred_enformer[test_info[i][2]] = {}
for i in range(len(test_info)):
    for it, track in enumerate(eval_tracks):
        final_pred_enformer[test_info[i][2]].setdefault(track, []).append(pred_matrix[it][i])

for i, gene in enumerate(final_pred_enformer.keys()):
    if i % 10 == 0:
        print(i, end=" ")
    for track in eval_tracks:
        final_pred_enformer[gene][track] = np.sum(final_pred_enformer[gene][track])
# OUR MODEL #####################################################################################################
#################################################################################################################
pred_matrix_our = joblib.load("pred_matrix_our.p")
print(f"{np.max(pred_matrix_our)}\t{np.std(pred_matrix_our)}\t{np.mean(pred_matrix_our)}\t{np.median(pred_matrix_our)}")
# heads = joblib.load(f"{p.pickle_folder}heads.gz")
# one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# # model_path = p.model_folder + "0.8070915954746051_0.5100707215128535/" + p.model_name 
# with strategy.scope():
#     our_model =mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads)
#     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
# pred_matrix_our = np.zeros((len(eval_tracks), len(test_info)))

# counts = [0, 0, 0, 0, 0]
# print("Predicting our")
# start = time.time()
# def get_seq(info):
#     start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
#     extra = start + p.input_size - len(one_hot[info[0]])
#     if start < 0:
#         ns = one_hot[info[0]][0:start + p.input_size]
#         ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
#     elif extra > 0:
#         ns = one_hot[info[0]][start: len(one_hot[info[0]])]
#         ns = np.concatenate((ns, np.zeros((extra, 5))))
#     else:
#         ns = one_hot[info[0]][start:start + p.input_size]
#     return ns[:, :-1]

# tss_index = 0
# for index, info in enumerate(test_info):
#     if index % 100 == 0:
#         print(index, end=" ")
#     batch = []
#     # for rvc in [True, False]:
#     for i in range(-1, 2, 1):
#         seq = get_seq([info[0], info[1] + i])
#     # seq = get_seq([info[0], info[1]])
#     #         if rvc:
#     #             seq = rev_comp(seq)
#         batch.append(seq)
#     batch = np.asarray(batch, dtype=bool)
#     pr = our_model.predict(mo.wrap2(batch, p.predict_batch_size))
#     p1 = np.mean(pr[1], axis=0)
#     for j, track in enumerate(eval_tracks):
#         t = track_ind_our[track]
#         bins = [p1[t, p.mid_bin - 1], p1[t, p.mid_bin], p1[t, p.mid_bin + 1]]
#         counts[bins.index(max(bins))] += 1 
#         pred_matrix_our[j, index] = np.sum(bins)

# print("")
# print(counts)
# end = time.time()
# print("Our time")
# print(end - start)
# joblib.dump(pred_matrix_our, "pred_matrix_our.p", compress=3)

final_pred_our = {}
for i in range(len(test_info)):
    final_pred_our[test_info[i][2]] = {}
for i in range(len(test_info)):
    for it, track in enumerate(eval_tracks):
        final_pred_our[test_info[i][2]].setdefault(track, []).append(pred_matrix_our[it][i])

for i, gene in enumerate(final_pred_our.keys()):
    if i % 10 == 0:
        print(i, end=" ")
    for track in eval_tracks:
        final_pred_our[gene][track] = np.sum(final_pred_our[gene][track])

# GT DATA #####################################################################################################
#################################################################################################################
load_info = []
for j, info in enumerate(test_info):
    mid = int(info[1] / p.bin_size)
    load_info.append([info[0], mid])
print("Loading ground truth tracks")
eval_track_names = []
for track in eval_tracks:
    eval_track_names.append(full_name[track])
gt_matrix = parser.par_load_data(load_info, eval_track_names, p).T
print(gt_matrix.shape)

eval_gt = {}
for i in range(len(test_info)):
    eval_gt[test_info[i][2]] = {}
for i in range(len(test_info)):
    for it, track in enumerate(eval_tracks):
        eval_gt[test_info[i][2]].setdefault(track, []).append(gt_matrix[it][i])

for i, gene in enumerate(eval_gt.keys()):
    if i % 10 == 0:
        print(i, end=" ")
    for track in eval_tracks:
        eval_gt[gene][track] = np.sum(eval_gt[gene][track])


print("")
def eval_perf(eval_gt, final_pred):
    scatter_data = []
    corr_s = []
    for gene in final_pred.keys():
        a = []
        b = []
        for v, track in enumerate(eval_tracks):
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.sum(b)==0:
            print(gene)
            continue
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
        else:
            corr_s.append(0)

    print("")
    print(f"Across tracks {len(corr_s)} {np.mean(corr_s)}")

    corr_s = []
    for track in eval_tracks:
        a = []
        b = []
        for gene in eval_gt.keys():
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.sum(b)==0:
            print(track)
            continue
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
        else:
            corr_s.append(0)
    print(f"Across genes {len(corr_s)} {np.mean(corr_s)}")

def eval_perf_norm(eval_gt, final_pred):
    corr_s = []
    for i in range(len(eval_gt[0])):
        a = final_pred[:, i]
        b = eval_gt[:, i]
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.sum(b)==0:
            print(gene)
            continue
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
        else:
            corr_s.append(0)

    print("")
    print(f"Across genes {len(corr_s)} {np.median(corr_s)}")
    scatter_data.append(corr_s)
    corr_s = []
    for i in range(len(eval_gt)):
        a = final_pred[i, :]
        b = eval_gt[i, :]
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        if np.sum(b)==0:
            print(track)
            continue
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc):
            corr_s.append(sc)
        else:
            corr_s.append(0)
    print(f"Across tracks {len(corr_s)} {np.median(corr_s)}")
    scatter_data.append(corr_s)
    return scatter_data

print("Enformer =================================================")
scatter_data_enf = eval_perf(eval_gt, final_pred_enformer)
print("OUR =================================================")
scatter_data_our = eval_perf(eval_gt, final_pred_our)


fig, axs = plt.subplots(1,2,figsize=(15, 5))
sns.scatterplot(x=scatter_data_our[0], y=scatter_data_enf[0], ax=axs[0])
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].plot([0, 0], [1, 1], linewidth=2, transform=axs[0].transAxes)
sns.scatterplot(x=scatter_data_our[1], y=scatter_data_enf[1], ax=axs[1])
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].plot([0, 0], [1, 1], linewidth=2, transform=axs[1].transAxes)
ax.set(xlabel='Predicted', ylabel='Ground truth')
plt.title("Gene expression prediction")
plt.tight_layout()
plt.savefig("predictions_scatter.svg")


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