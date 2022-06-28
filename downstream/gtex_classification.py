import pandas as pd
import pathlib
import joblib
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import attribution
# import logomaker
from sklearn.cluster import KMeans
import seaborn as sns
import model as mo
from main_params import MainParams
import common as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

TRACK_NUM = 40
DISEASE_NUM = 200
VCF_DIR = "data/gtex_pos_neg/"


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


p = MainParams()

head_name = "hg38"
heads = joblib.load(f"{p.pickle_folder}heads.gz")
head = heads[head_name]
one_hot = joblib.load(f"{p.pickle_folder}{head_name}_one_hot.gz")

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, head, use_hic=False)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_" + head_name))
    our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
    our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))

vcf_names = set([])
for filename in sorted(os.listdir(VCF_DIR)):
    name = filename[:-len("_pos.vcf")]
    vcf_names.add(name)

vcf_names = sorted(list(vcf_names))
print(f"Num vcf: {len(vcf_names)}")
AUCs = []
for name in vcf_names:
    print(name)
    pos = pd.read_csv(VCF_DIR + name + "_pos.vcf", sep="\t", index_col=False,
                      names=["chrom", "position", "info", "ref", "alt", "c1", "c2"], comment="#")
    pos = pos.drop(pos[pos.position < p.half_size - 1].index)
    neg = pd.read_csv(VCF_DIR + name + "_neg.vcf", sep="\t", index_col=False,
                      names=["chrom", "position", "info", "ref", "alt", "c1", "c2"], comment="#")
    neg = neg.drop(neg[neg.position < p.half_size - 1].index)
    Y_label_orig = np.concatenate([np.ones_like(np.arange(len(pos))), np.zeros_like(np.arange(len(neg)))], axis=0)
    df = pos.append(neg, ignore_index=True)
    seqs1 = []
    seqs2 = []
    Y_label = []
    for index, row in df.iterrows():
        start = row["position"] - p.half_size - 1
        if row["chrom"] not in one_hot.keys():
            continue

        correction = 0
        if start < 0:
            correction = start
            start = 0
        if start + p.input_size > len(one_hot[row["chrom"]]):
            extra = (start + p.input_size) - len(one_hot[row["chrom"]])
            start = start - extra
            correction = extra

        Y_label.append(Y_label_orig[index])
        snp_pos = p.half_size + correction
        seq1 = one_hot[row["chrom"]][start: start + p.input_size][..., :-1]
        ref = seq1[snp_pos][["ACGT".index(row["ref"])]] # True
        alt1 = seq1[snp_pos][["ACGT".index(row["alt"])]] # False
        seqs1.append(seq1)
        seq2 = seq1.copy()
        seq2[snp_pos] = [False, False, False, False]
        seq2[snp_pos][["ACGT".index(row["alt"])]] = True
        alt2 = seq2[snp_pos][["ACGT".index(row["alt"])]] # True
        seqs2.append(seq2)

    print(f"Predicting {len(seqs1)}")
    dif = mo.batch_predict_effect(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
    print("Done")
    

    X_train, X_test, Y_train, Y_test = train_test_split(dif, np.asarray(Y_label), test_size=0.1, random_state=1)
    clf = RandomForestClassifier(random_state=0) # max_depth=100,
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
    auc1 = metrics.auc(fpr, tpr)
    print(f"{name} AUC: {auc1}")


    pca = PCA(n_components=100)
    pca_data = pca.fit_transform(dif)
    X_train, X_test, Y_train, Y_test = train_test_split(pca_data, np.asarray(Y_label), test_size=0.1, random_state=1)
    clf = RandomForestClassifier(random_state=0) # max_depth=100,
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
    auc2 = metrics.auc(fpr, tpr)
    print(f"{name} AUC: {auc2}")
    AUCs.append(max(auc1, auc2))


print(f"Average AUC: {np.mean(AUCs)}")
