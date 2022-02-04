import os
import pathlib
from skimage import measure
import model as mo
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import attribution
import logomaker
from sklearn.cluster import KMeans
import seaborn as sns
from main_params import MainParams
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


p = MainParams()
script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])
parsed_tracks_folder = folders[1]
parsed_hic_folder = folders[2]
model_folder = folders[3]
heads = joblib.load("pickle/heads.gz")
head_id = 0
head_tracks = heads[head_id]
one_hot = joblib.load("pickle/one_hot.gz")
gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
tss_loc = joblib.load("pickle/tss_loc.gz")
for key in tss_loc.keys():
    tss_loc[key].sort()

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = tf.keras.models.load_model(model_folder + p.model_name,
                                           custom_objects={'PatchEncoder': mo.PatchEncoder})
    our_model.get_layer("our_head").set_weights(joblib.load(model_folder + p.model_name + "_head_" + str(head_id)))

bed_dir = "/home/user/data/hg38_bed/"
track_names = []
disease_names = []
heatmap_matrix = []
for filename in sorted(os.listdir(bed_dir)):
    disease_names.append(filename)
    snp_pos = pd.read_csv(bed_dir + filename, sep="\t", index_col=False,
                                   names=["chrom", "start", "end", "info", "score"])
    maes = []
    for track_to_use, track in enumerate(head_tracks):
        type = track[:track.find(".")]
        if type != "scEnd5":
            continue
        # one time only
        if len(heatmap_matrix) == 0:
            track_names.append(track)
        seqs1 = []
        seqs2 = []
        for index, row in snp_pos.iterrows():
            start = row["start"] - p.half_size - 1 # 1 based, right?
            seq1 = one_hot[row["chrom"]][start: start + p.input_size]
            seqs1.append(seq1)
            seq2 = seq1.copy()
            seq2[p.mid_bin] = [False, False, False, False, False]
            seqs2.append(seq2)
        vals1 = our_model.predict(seqs1)
        vals2 = our_model.predict(seqs2)
        mae = np.sum(np.absolute((vals1 - vals2)))
        maes.append(mae)
    heatmap_matrix.append(maes)
    if len(heatmap_matrix) > 10:
        continue

heatmap_matrix = np.asarray(heatmap_matrix)
heatmap_matrix = pd.DataFrame(data=heatmap_matrix, index=disease_names, columns=track_names)
fig, ax = plt.subplots(figsize=(16, 6))
g = sns.clustermap(heatmap_matrix)
plt.savefig(f"temp/clustermap.png")
plt.close(fig)
