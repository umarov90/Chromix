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

# model_path = p.model_folder + "0.8070915954746051_0.5100707215128535/" + p.model_name 
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))

vcf_names = set([])
for filename in sorted(os.listdir(VCF_DIR)):
    name = filename[:-len("_pos.vcf")]
    vcf_names.add(name)

vcf_names = sorted(list(vcf_names))
# Number of files to use!
vcf_names = vcf_names[6:12] 
AUCs = []
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
    #############################################################
    print(f"Predicting {len(seqs1)}")
    # Turn the two outputs into feature vectors for Random Forests
    dif, fold_changes = mo.batch_predict_effect(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
    # dif = calculate_effect_pca(np.asarray(seqs1), np.asarray(seqs2))
    print("Done")
    

    X_train, X_test, Y_train, Y_test = train_test_split(dif, np.asarray(Y_label), test_size=0.3, random_state=1)

    pca = PCA(n_components=20)
    pca.fit(X_train)

    def transform(xx):
        xx = pca.transform(xx)
        return xx

    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict_proba(X_test)[:,1]
    auc1 = roc_auc_score(Y_test, Y_pred)
    print(f"{name} AUC1: {auc1}")
    # AUCs.append(auc1)

    X_train = transform(X_train)
    X_test = transform(X_test)
    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict_proba(X_test)[:,1]
    auc2 = roc_auc_score(Y_test, Y_pred)
    print(f"{name} AUC2: {auc2}")

    AUCs.append(max(auc1, auc2))

    # fig, axs = plt.subplots(1,1,figsize=(15, 15))
    # RocCurveDisplay.from_estimator(clf, X_test, Y_test, ax=axs, name=name)
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate") 
    # plt.tight_layout()
    # plt.savefig(name + "_roc_pca.png")



# We used principal component analysis (PCA) to reduce 5,313 variant effect features from Enformer to 20 principle components.
# We used variant effect scores from 1000 Genomes SNPs on chromosome 9 and performed the following steps:
# (1) subtracted the median and divided by standard deviation estimated from the interquartile range as implemented in RobustScaler in scikit-learn (v0.23.2);
# (2) reduced the dimensionality to 20 principle components using TruncatedSVD from scikit-learn; and 
# (3) normalized the resulting principal component features using RobustScaler to obtain z-scores.



# Turn off dropout!!!!!!!!!! or multiple eval

    


print(f"Average AUC: {np.mean(AUCs)}")
