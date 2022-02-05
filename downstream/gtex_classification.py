import pandas as pd
import pathlib
import joblib
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import attribution
import logomaker
from sklearn.cluster import KMeans
import seaborn as sns
import model as mo
from main_params import MainParams
import common as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

TRACK_NUM = 40
DISEASE_NUM = 200
VCF_DIR = "data/gtex_pos_neg/"


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

vcf_names = set([])
for filename in sorted(os.listdir(VCF_DIR)):
    name = filename[:-len("_pos.vcf")]
    vcf_names.add(name)

vcf_names = sorted(list(vcf_names))

for name in vcf_names:
    pos = pd.read_csv(VCF_DIR + name + "_pos.vcf", sep="\t", index_col=False,
                      names=["chrom", "position", "info", "ref", "alt", "c1", "c2"], comment="#")
    pos = pos.drop(pos[pos.position < p.half_size - 1].index)
    neg = pd.read_csv(VCF_DIR + name + "_neg.vcf", sep="\t", index_col=False,
                      names=["chrom", "position", "info", "ref", "alt", "c1", "c2"], comment="#")
    neg = neg.drop(neg[neg.position < p.half_size - 1].index)
    Y_label = np.concatenate([np.ones_like(np.arange(len(pos))), np.zeros_like(np.arange(len(neg)))], axis=0)
    df = pos.append(neg, ignore_index=True)
    seqs1 = []
    seqs2 = []
    for index, row in df.iterrows():
        start = row["position"] - p.half_size - 1
        seq1 = one_hot[row["chrom"]][start: start + p.input_size]
        ref = seq1[p.half_size][["ACGT".index(row["ref"])]] # True
        alt1 = seq1[p.half_size][["ACGT".index(row["alt"])]] # False
        seqs1.append(seq1)
        seq2 = seq1.copy()
        seq2[p.half_size] = [False, False, False, False, seq2[p.half_size][4]]
        seq2[p.half_size][["ACGT".index(row["alt"])]] = True
        alt2 = seq2[p.half_size][["ACGT".index(row["alt"])]] # True
        seqs2.append(seq2)

    vals1 = our_model.predict(mo.wrap2(np.asarray(seqs1), 16))
    vals2 = our_model.predict(mo.wrap2(np.asarray(seqs2), 16))
    dif = np.mean(vals1 - vals2, axis=-1)

    X_train, X_test, Y_train, Y_test = train_test_split(dif, Y_label, test_size=0.1, random_state=1)
    clf = RandomForestClassifier(max_depth=100, random_state=0)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(f"{name} AUC: {auc}")
