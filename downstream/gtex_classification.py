import pandas as pd
import tensorflow as tf
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
sns.set(font_scale = 2.5)
import model as mo
from main_params import MainParams
import common as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import RocCurveDisplay
from enformer_usage import calculate_effect, calculate_effect_pca
from sklearn.metrics import roc_auc_score
import enhancers.enhancer_linking as el
import parse_data as parser


VCF_DIR = "data/gtex_pos_neg/"


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


p = MainParams()
input_size = p.input_size
# input_size = 393216
half_size = input_size // 2

head = joblib.load(f"{p.pickle_folder}heads.gz")
one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")
hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]

train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
infos = train_info + valid_info + test_info
tss_dict = {}
for info in infos:
    tss_dict.setdefault(info[0], []).append(info[1])

for key in tss_dict.keys():
    tss_dict[key].sort()

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))

vcf_names = []
for filename in sorted(os.listdir(VCF_DIR)):
    if filename.endswith("_pos.vcf"):        
        name = filename[:-len("_pos.vcf")]
        vcf_names.append(name)

vcf_names.sort()
# Number of files to use!
vcf_names = vcf_names[6:12] 
AUCs = {}
print(f"Num vcf: {len(vcf_names)}")
data = {}
for name in vcf_names:
    # get positive and negative examples
    print(name)
    pos = pd.read_csv(VCF_DIR + name + "_pos.vcf", sep="\t", index_col=False,
                      names=["chrom", "position", "info", "ref", "alt", "c1", "c2"], comment="#")
    pos = pos.drop(pos[pos.position < half_size - 1].index)
    neg = pd.read_csv(VCF_DIR + name + "_neg.vcf", sep="\t", index_col=False,
                      names=["chrom", "position", "info", "ref", "alt", "c1", "c2"], comment="#")
    neg = neg.drop(neg[neg.position < half_size - 1].index)
    print(f"{len(pos)} - {len(neg)}")
    Y_label_orig = np.concatenate([np.ones_like(np.arange(len(pos))), np.zeros_like(np.arange(len(neg)))], axis=0)
    df = pos.append(neg, ignore_index=True)
    seqs1 = []
    seqs2 = []
    Y_label = []
    enhancer_scores = []
    bn_enhancer_scores = []
    for index, row in df.iterrows():
        start = row["position"] - half_size - 1
        if row["chrom"] not in one_hot.keys():
            continue

        extra = start + input_size - len(one_hot[row["chrom"]])
        if start < 0:
            ns = one_hot[row["chrom"]][0:start + input_size]
            ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
        elif extra > 0:
            ns = one_hot[row["chrom"]][start: len(one_hot[row["chrom"]])]
            ns = np.concatenate((ns, np.zeros((extra, 5))))
        else:
            ns = one_hot[row["chrom"]][start:start + input_size]

        Y_label.append(Y_label_orig[index])
        snp_pos = half_size
        seq1 = ns[..., :-1]
        ref = seq1[snp_pos][["ACGT".index(row["ref"])]] # True
        alt1 = seq1[snp_pos][["ACGT".index(row["alt"])]] # False
        seqs1.append(seq1)
        seq2 = seq1.copy()
        seq2[snp_pos] = [False, False, False, False]
        seq2[snp_pos][["ACGT".index(row["alt"])]] = True
        alt2 = seq2[snp_pos][["ACGT".index(row["alt"])]] # True
        seqs2.append(seq2)
        enhancer_score = 0
        tss_candidates = [x for x in tss_dict[row["chrom"]] if row["position"] - (half_size - 2000) <= x <= row["position"] + (half_size - 2000)]
        for t in tss_candidates:
            if abs(row["position"] - t) < 2000:
                enhancer_score = 1.01
                break
        if len(tss_candidates) > 0 and enhancer_score == 0:
            scores = el.linking_proba(row["chrom"], tss_candidates, [row["position"]]*len(tss_candidates), one_hot, our_model)
            enhancer_score = max(scores)
        enhancer_scores.append(enhancer_score)
        bn_enhancer_scores.append(enhancer_score == 1.01)
        print(f"TSS candidates: {len(tss_candidates)}. Max enhancer score: {enhancer_score}")

    enhancer_scores = np.expand_dims(np.asarray(enhancer_scores), axis=1)
    bn_enhancer_scores = np.expand_dims(np.asarray(bn_enhancer_scores), axis=1)
    joblib.dump(enhancer_scores, "enhancer_scores.p", compress=3)
    # enhancer_scores = joblib.load("enhancer_scores.p")
    print("===========================================================")
    print(enhancer_scores.shape)
    print(bn_enhancer_scores.shape)
    #############################################################
    print(f"Predicting {len(seqs1)}")
    # Turn the two outputs into feature vectors for Random Forests
    dif, fold_changes = mo.batch_predict_effect_x(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
    joblib.dump(dif, "dif_gtex.p", compress=3)
    # dif = joblib.load("dif_gtex.p")
    # dif = calculate_effect_pca(np.asarray(seqs1), np.asarray(seqs2))
    print("Done")
    for method in ["Baseline", "Binary score", "Enhancer score"]:
        print(method)
        if method != "Baseline":
            if method == "Binary score":
                add_features = bn_enhancer_scores
            else:
                add_features = enhancer_scores
            dif = np.concatenate((dif, add_features), axis=-1)

        X_train, X_test, Y_train, Y_test = train_test_split(dif, np.asarray(Y_label), test_size=0.3, random_state=1)
        if method != "Baseline":
            X_train_dist = X_train[:, -1 * add_features.shape[1]:]
            X_test_dist = X_test[:, -1 * add_features.shape[1]:]
            X_train = X_train[:, :-1 * add_features.shape[1]]
            X_test = X_test[:, :-1 * add_features.shape[1]]

        pca = PCA(n_components=10)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        if method != "Baseline":
            X_train = np.concatenate((X_train, X_train_dist), axis=-1)
            X_test = np.concatenate((X_test, X_test_dist), axis=-1)

        clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(Y_test, Y_pred)
        print(f"{name} AUC: {auc}")

        AUCs.setdefault(method, []).append(auc)

    # fig, axs = plt.subplots(1,1,figsize=(15, 15))
    # RocCurveDisplay.from_estimator(clf, X_test, Y_test, ax=axs, name=name)
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate") 
    # plt.tight_layout()
    # plt.savefig(name + "_roc_pca.png")
for method in AUCs.keys():
    print(f"{method} average AUC: {np.mean(AUCs[method])}")

df = pd.DataFrame.from_dict(AUCs)
df.to_csv('gtex_auc.csv', index=False)