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
import common as cm
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

TRACK_NUM = 40
DISEASE_NUM = 200


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

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = tf.keras.models.load_model(model_folder + p.model_name,
                                           custom_objects={'PatchEncoder': mo.PatchEncoder})
    our_model.get_layer("our_head").set_weights(joblib.load(model_folder + p.model_name + "_head_" + str(head_id)))

bed_dir = "/home/user/data/hg38_bed/"
track_names = []
disease_names = []
heatmap_matrix = []
for i, filename in enumerate(sorted(os.listdir(bed_dir))):
    disease_name = filename[cm.find_nth(filename, '.', 2) + 1: cm.find_nth(filename, '.', 3)]
    disease_names.append(disease_name)
    snp_pos = pd.read_csv(bed_dir + filename, sep="\t", index_col=False,
                                   names=["chrom", "start", "end", "info", "score"])

    seqs1 = []
    seqs2 = []
    for index, row in snp_pos.iterrows():
        start = row["start"] - p.half_size - 1 # 1 based, right?
        seq1 = one_hot[row["chrom"]][start: start + p.input_size]
        # one_hot[row["chrom"]][row["start"] - 1: row["start"]]
        seqs1.append(seq1)
        for j in range(4):
            if seq1[p.half_size][j]:
                continue
            seq2 = seq1.copy()
            seq2[p.half_size] = [False, False, False, False, seq2[p.half_size][4]]
            seq2[p.half_size][j] = True
            seqs2.append(seq2)
    vals1 = our_model.predict(mo.wrap2(np.asarray(seqs1), 16))
    vals2 = our_model.predict(mo.wrap2(np.asarray(seqs2), 16))

    maes = []
    for track_to_use, track in enumerate(head_tracks):
        type = track[:track.find(".")]
        if type != "scEnd5":
            continue
        # one time only
        if len(heatmap_matrix) == 0:
            track_names.append(track[cm.find_nth(track, '.', 3) + 1: cm.find_nth(track, '.', 5)])
        # mae = np.sum(np.absolute((vals1[:, track_to_use, :] - vals2[:, track_to_use, :])))
        mae = 0
        for a in range(len(seqs1)):
            mae1 = np.sum(np.absolute((vals1[a, track_to_use, :] - vals2[3 * a, track_to_use, :])))
            mae2 = np.sum(np.absolute((vals1[a, track_to_use, :] - vals2[3 * a + 1, track_to_use, :])))
            mae3 = np.sum(np.absolute((vals1[a, track_to_use, :] - vals2[3 * a + 2, track_to_use, :])))
            mae += max(mae1, mae2, mae3)
        maes.append(mae)
        if len(maes) >= TRACK_NUM:
            break

    heatmap_matrix.append(maes)
    print(f"{i} : {disease_name}")
    if len(heatmap_matrix) > DISEASE_NUM:
        break
heatmap_matrix = np.asarray(heatmap_matrix)
joblib.dump(heatmap_matrix, "temp/heat.p")

heatmap_matrix = joblib.load("temp/heat.p")
heatmap_matrix = pd.DataFrame(data=heatmap_matrix, index=disease_names, columns=track_names)
g = sns.clustermap(heatmap_matrix, rasterized=True, figsize=(16, 5*16), yticklabels=True, xticklabels=True)
plt.savefig(f"temp/clustermap.svg")
