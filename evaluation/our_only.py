import gc
import os
import pickle
import multiprocessing as mp
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import joblib
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
from scipy import stats
import pathlib
import pyBigWig
import model as mo
from main_params import MainParams

fasta_file = '/Users/ramzan/variants_100/data/species/hg38/genome.fa'

targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')

script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])
parsed_tracks_folder = folders[1]
parsed_hic_folder = folders[2]
model_folder = folders[3]
params = MainParams()

gene_tss = pd.read_csv("data/enformer/enformer_test_tss_less.bed", sep="\t", index_col=False,
                       names=["chr", "start", "end", "type"])
# gene_tss = gene_tss.loc[gene_tss['chr'] == "chr1"]
# gene_tss.index = range(len(gene_tss.index))
gene_tss = gene_tss.head(2)
# eval_tracks = pd.read_csv("data/eval_tracks.tsv", sep=",", header=None)[0].tolist()
enf_tracks = df_targets[df_targets['description'].str.contains("CAGE")]['identifier'].tolist()
# extract from enformer targets identifier where row contains cage
# full_name = pickle.load(open(f"pickle/full_name.p", "rb"))
heads = joblib.load("pickle/heads.gz")
head_id = 0
head_name = "hg38"
head_tracks = heads[head_id]
track_ind_our = {}
eval_tracks = []
for track in enf_tracks:
    for our_track in head_tracks:
        if track in our_track:
            track_ind_our[track] = head_tracks.index(our_track)
            eval_tracks.append(track)
            break

full_name = {}
tracks_folder = "/Volumes/passport1/bw/"
for filename in os.listdir(tracks_folder):
    for track in eval_tracks:
        if track in filename:
            size = os.path.getsize(tracks_folder + filename)
            full_name[track] = filename
# pickle.dump(full_name, open(f"pickle/full_name.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
# np.savetxt("data/eval_tracks.tsv", eval_tracks, delimiter="\t", fmt='% s')
print(f"Eval tracks {len(eval_tracks)}")
print(f"TSS {len(gene_tss)}")
# track_ind = pickle.load(open(f"pickle/track_ind.p", "rb"))
track_ind = {}
for j, track in enumerate(eval_tracks):
    track_ind[track] = df_targets[df_targets['identifier'].str.contains(track)].index

# pickle.dump(track_ind, open(f"pickle/track_ind.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# OUR MODEL #####################################################################################################
#################################################################################################################
one_hot = joblib.load("pickle/hg38_one_hot.gz")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.small_model(params.input_size, params.num_features, params.num_bins, len(head_tracks), params.bin_size)
    print(our_model.summary())
    our_model.get_layer("our_resnet").set_weights(joblib.load(params.model_path + "_res"))
    our_model.get_layer("our_head").set_weights(joblib.load(params.model_path + "_head_" + head_name))
pred_matrix_our = np.zeros((len(eval_tracks), len(gene_tss)))
p = MainParams()
test_seq = []
predict_batch_size = p.GLOBAL_BATCH_SIZE
w_step = 500
counts = [0, 0, 0]
print("Predicting our")
gene_index = 0
for index, row in gene_tss.iterrows():
    start = int(row["start"] - p.half_size)
    ns = one_hot[row["chr"]][start:start + p.input_size, :-1]
    test_seq.append(ns)

for w in range(0, len(test_seq), w_step):
    print(w, end=" ")
    dt = mo.wrap2(test_seq[w:w + w_step], predict_batch_size)
    p1 = our_model.predict(dt)
    for m in range(len(p1)):
        for j, track in enumerate(eval_tracks):
            t = track_ind_our[track]
            bins = [p1[m, t, p.mid_bin - 1], p1[m, t, p.mid_bin], p1[m, t, p.mid_bin + 1]]
            counts[bins.index(max(bins))] += 1
    p2 = p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
    if w == 0:
        predictions = p2
    else:
        predictions = np.concatenate((predictions, p2), dtype=np.float32)

for i in range(len(gene_tss)):
    for j, track in enumerate(eval_tracks):
        t = track_ind_our[track]
        pred_matrix_our[j, i] = predictions[i, t]

print("")
print(counts)

# GT DATA #####################################################################################################
#################################################################################################################

gt_matrix = np.zeros((len(eval_tracks), len(gene_tss)))

bin = 200 # 128
half_bin = 100 # 63

# for track in eval_tracks:
#     gt_vector = np.zeros(len(gene_tss))
#     bw = pyBigWig.open(tracks_folder + full_name[track])
#     for index, row in gene_tss.iterrows():
#         start = row["start"] - half_bin - bin
#         vals = bw.values(row["chr"], start, start + 3 * bin)
#         vals = np.nan_to_num(vals)
#         vals = [np.sum(vals[0:bin]), np.sum(vals[bin:2 * bin]), np.sum(vals[2 * bin:])]
#         gt_count = 0
#         for v in vals:
#             if v > 0:
#                 gt_count += np.log10(v)
#         gt_vector[index] = gt_count
#     bw.close()
#     joblib.dump(gt_vector, "/media/user/EE3C38483C380DD9/temp/gt_" + str(eval_tracks.index(track)))
#     print(track)


def load_bw(q, sub_tracks):
    for track in sub_tracks:
        gt_vector = np.zeros(len(gene_tss))
        bw = pyBigWig.open(tracks_folder + full_name[track])
        for index, row in gene_tss.iterrows():
            start = row["start"] - half_bin - bin
            vals = bw.values(row["chr"], start, start + 3 * bin)
            vals = np.nan_to_num(vals)
            vals = np.asarray([np.sum(vals[0:bin]), np.sum(vals[bin:2 * bin]), np.sum(vals[2 * bin:])])
            gt_vector[index] = np.sum(vals)

            # start = row["start"] - 64
            # vals = bw.values(row["chr"], start, start + 128)
            # vals = np.nan_to_num(vals)
            # gt_vector[index] = np.sum(vals)
        bw.close()
        joblib.dump(gt_vector, "temp/gt_"+str(eval_tracks.index(track)))
        print(track)
    q.put(None)


print("Loading GT")
step_size = 30
q = mp.Queue()
ps = []
start = 0
nproc = 10
end = len(eval_tracks)
for t in range(start, end, step_size):
    t_end = min(t+step_size, end)
    sub_tracks = eval_tracks[t:t_end]
    p = mp.Process(target=load_bw,
                   args=(q, sub_tracks,))
    p.start()
    ps.append(p)
    if len(ps) >= nproc:
        for p in ps:
            p.join()
        print(q.get())
        ps = []

if len(ps) > 0:
    for p in ps:
        p.join()
    print(q.get())

print("")
for i in range(len(eval_tracks)):
    gt_matrix[i, :] = joblib.load("temp/gt_"+str(i))

print("OUR =================================================")
corrs = []
for i in range(len(eval_tracks)):
    a = pred_matrix_our[i, :]
    b = gt_matrix[i, :]
    corr = stats.spearmanr(a, b)[0]
    corrs.append(corr)

print(f"Across tracks {np.median(np.nan_to_num(corrs))}")

corrs = []
for i in range(len(gene_tss)):
    a = pred_matrix_our[:, i]
    b = gt_matrix[:, i]
    corr = stats.spearmanr(a, b)[0]
    corrs.append(corr)

print(f"Across genes {np.median(np.nan_to_num(corrs))}")


