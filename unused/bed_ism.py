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
model_name = "small.h5"
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
print(our_model.get_layer("our_transformer").get_layer("pos_embedding").summary())
exit()
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
        for nuc in range(4):
            nsc = ns.copy()
            a = np.zeros(num_features)
            a[num_features - 1] = ns[mut_pos][num_features - 1]
            a[nuc] = 1
            nsc[mut_pos] = a
            sequences.append(nsc)

        nsc = ns.copy()
        nsc[mut_pos][num_features - 1] = 0
        sequences.append(nsc)

        preds = our_model.predict(np.asarray(sequences))[0]
        difs = []
        for i in range(len(preds)):
            difs.append(preds[i][track_to_use] - preds[orig][track_to_use])

        fig, axs = plt.subplots(5, 1, figsize=(15,15), sharex=True)
        letters = ["A", "C", "G", "T", "5"]
        for i in range(5):
            y = difs[i]
            x = range(len(y))
            axs[i].fill_between(x, y)
            axs[i].set_title(letters[i])

        plt.savefig(f"temp/{chrn}.{chrp}.png")
        plt.close(fig)



        exit()
        # max nucl
        # max tss
        # plot 3 substracted vectors