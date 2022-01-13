import os
import pathlib
import model as mo
import tensorflow as tf
from tensorflow.python.keras import backend as K
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import attribution
import seaborn as sns
import skimage.measure as measure
import math
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


input_size = 210001 # 50001
half_size = int(input_size / 2)
bin_size = 200
num_features = 5
head_id = 0
track_to_use = 5
num_regions = 501
mid_bin = math.floor(num_regions / 2)
script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/data_dirs").read().strip().split("\n")
os.chdir(folders[0])
parsed_tracks_folder = folders[1]
parsed_hic_folder = folders[2]
model_folder = folders[3]
model_name = "all_tracks.h5"
heads = joblib.load("pickle/heads.gz")
head_tracks = heads[head_id]
one_hot = joblib.load("pickle/one_hot.gz")
test_info = joblib.load("pickle/test_info.gz")
train_info = joblib.load("pickle/train_info.gz")
all_info = train_info.extend(test_info)
gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
tss_loc = joblib.load("pickle/tss_loc.gz")
for key in tss_loc.keys():
    tss_loc[key].sort()


our_model = tf.keras.models.load_model(model_folder + model_name,
                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
our_model.get_layer("our_head").set_weights(joblib.load(model_folder + model_name + "_head_" + str(head_id)))

difs_total = []
with open('data/hg38.GENCODEv38.pc_lnc.TSS.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        chrp = int(vals[1])
        start = int(chrp - half_size)
        extra = start + input_size - len(one_hot[chrn])
        if start < 0:
            ns = one_hot[chrn][0:start + input_size]
            ns = np.concatenate((np.zeros((-1 * start, num_features)), ns))
        elif extra > 0:
            ns = one_hot[chrn][start: len(one_hot[chrn])]
            ns = np.concatenate((ns, np.zeros((extra, num_features))))
        else:
            ns = one_hot[chrn][start:start + input_size]

        mut_pos = half_size
        orig = np.where(ns[mut_pos][:4] == 1)[0][0]
        sequences = []
        sequences.append(ns)
        step = 20
        r = int(2000 / step)
        for i in range(r):
            nsc = ns.copy()
            a = np.zeros((step, num_features))
            nsc[mut_pos - 1000 + i * step: mut_pos - 1000 + (i + 1) * step] = a
            sequences.append(nsc)

        preds = our_model.predict(np.asarray(sequences), batch_size=8)[0]
        difs = []
        for i in range(1, len(preds), 1):
            difs.append(preds[i][track_to_use][mid_bin] - preds[0][track_to_use][mid_bin])
        difs_total.append(difs)
        print(len(difs_total))
        if len(difs_total) > 100:
            break

difs_total = np.mean(np.asarray(difs_total), axis=0)

fig = plt.figure(figsize=(15,10))
plt.fill_between(range(len(difs_total)), difs_total)

plt.savefig(f"temp/100genes.svg")
plt.close(fig)