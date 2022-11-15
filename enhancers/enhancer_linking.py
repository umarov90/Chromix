import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import model as mo
import pandas as pd
import pathlib
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
sns.set(font_scale = 2.5)
from main_params import MainParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import umap
from scipy import stats
from sklearn.metrics import RocCurveDisplay
from bisect import bisect_left
import parse_data as parser
from random import randrange
from collections import Counter
from sklearn.metrics import roc_auc_score
import random
from enformer_usage import calculate_effect
from scipy import stats


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_linking_AUC():
    p = MainParams()

    head = joblib.load(f"{p.pickle_folder}heads.gz")
    one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")

    # strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    #     our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head["expression"])
    #     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
        # # our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))

    train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
    infos = train_info + valid_info + test_info
    tss_dict = {}
    for info in infos:
        tss_dict.setdefault(info[0], []).append(info[1])

    for key in tss_dict.keys():
        tss_dict[key].sort()

    df = pd.read_csv("data/enhancers/all.tsv", sep="\t")
    # df = df.sample(frac=0.1)

    def add_seqs(tss_pos, enh_mid, input_size, half_size):
        start = tss_pos - half_size - 1
        if start < 0 or start + input_size > len(one_hot[row["chr"]]):
            return None, None, None
        relative1 = tss_pos - enh_mid
        if abs(relative1) > half_size or abs(relative1) < 2000: 
            return None, None, None
        enh_pos = half_size - relative1
        fseq1 = one_hot[row["chr"]][start: start + input_size][..., :-1]     
        fseq2 = fseq1.copy()
        fseq2[enh_pos - 1000: enh_pos + 1000] = 0
        return fseq1, fseq2, relative1

    seqs1 = []
    seqs2 = []
    eseqs1 = []
    eseqs2 = []
    Y_label = []
    region_type = []
    distances = []
    enhancer_pos = []
    picked_regions_hic = []
    tss_pos = []
    for index, row in df.iterrows():    
        if row["chr"] not in one_hot.keys():
            continue
        seq1, seq2, relative = add_seqs(row["Gene TSS"], row["mid"], p.input_size, p.half_size)
        eseq1, eseq2, erelative = add_seqs(row["Gene TSS"], row["mid"], 393216, 393216 // 2)
        if relative is not None and erelative is not None: 
            Y_label.append(row['Significant'])
            distances.append(abs(relative))
            region_type.append(row['Significant'])
            enhancer_pos.append([row["chr"], row["mid"] // p.bin_size])
            picked_regions_hic.append([row["chr"], min(row["mid"], row["Gene TSS"]), max(row["mid"], row["Gene TSS"])])
            tss_pos.append([row["chr"], row["Gene TSS"] // p.bin_size])
            seqs1.append(seq1)
            seqs2.append(seq2)

            eseqs1.append(eseq1)
            eseqs2.append(eseq2)

    print(Counter(region_type))

    print(f"Predicting {len(seqs1)}")
    # dif, fold_changes = mo.batch_predict_effect(p, our_model, np.asarray(seqs1), np.asarray(seqs2))

    # enformer_effect = calculate_effect(np.asarray(eseqs1), np.asarray(eseqs2))
    # dif = enformer_effect
    # joblib.dump(dif, "enformer_effect.p", compress=3)
    # dif = joblib.load("enformer_effect.p")

    # print("Done")
    # joblib.dump([dif, fold_changes], "enhancer_rf.p", compress=3)
    # print("Dumped")
    dump = joblib.load("enhancer_rf.p")
    dif = dump[0]
    fold_changes = dump[1]
    print(f"Effects {dif.shape} {len(Y_label)}")

    marks = ["h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3", "h3k36me3", "h3k27ac"]
    track_types = ["CAGE", "ATAC", "h3k27ac"]
    # marks = ["h3k27ac"]
    def get_signals(positions, head, kw, do_mean=True):
        signals = None
        for i, mark in enumerate(kw):
            chosen_tracks1 = []
            for track in head:
                # if "K562" in track:
                if mark in track.lower():
                    chosen_tracks1.append(track)
            # print(f"{len(chosen_tracks1)} {mark} tracks found")
            signal = parser.par_load_data_temp(positions, chosen_tracks1, p)
            if do_mean:
                signal = np.mean(signal, axis=-1, keepdims=True)
            if signals is None:
                signals = signal
            else:
                signals = np.concatenate((signals, signal), axis=1)
        return signals

    signals = get_signals(enhancer_pos, head["epigenome"], marks)
    signals_tss = get_signals(tss_pos, head["epigenome"], marks)
    print(f"Signals {signals.shape}")

    # def compute_corrs(a, b):
    #     corrs = []
    #     for i in range(len(a)):
    #         corrs.append(stats.spearmanr(a[i], b[i])[0])
    #     corrs = np.expand_dims(np.asarray(corrs), axis=1)
    #     corrs[np.isnan(corrs)] = 0
    #     return corrs
    #
    # def sanity_check(some_signal, name):
    #     p = []
    #     n = []
    #     for i in range(len(Y_label)):
    #         if Y_label[i]:
    #             p.append(some_signal[i])
    #         else:
    #             n.append(some_signal[i])
    #     print(f"{name} Avg p {np.mean(p)} Avg n {np.mean(n)}")
    #
    # signals_enh_cage = get_signals(enhancer_pos, head["expression"], ["cage."], False)
    # signals_tss_cage = get_signals(tss_pos, head["expression"], ["cage."], False)
    # cage_corr = compute_corrs(signals_enh_cage, signals_tss_cage)
    # sanity_check(cage_corr, "CAGE corr")
    #
    # signals_enh_atac = get_signals(enhancer_pos, head["epigenome"], ["atac."], False)
    # signals_tss_atac = get_signals(tss_pos, head["epigenome"], ["atac."], False)
    # atac_corr = compute_corrs(signals_enh_atac, signals_tss_atac)
    # sanity_check(atac_corr, "atac_corr")
    #
    # signals_enh_ac = get_signals(enhancer_pos, head["epigenome"], ["h3k27ac"], False)
    # signals_tss_ac = get_signals(tss_pos, head["epigenome"], ["h3k27ac"], False)
    # ac_corr = compute_corrs(signals_enh_ac, signals_tss_ac)
    # sanity_check(ac_corr, "ac_corr")

    hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    hic_signal = parser.par_load_hic_data_one(hic_keys, p, picked_regions_hic)
    hic_signal[np.isnan(hic_signal)] = 0
    hic_signal = np.expand_dims(hic_signal, axis=1)
    print(f"HiC signal shape {hic_signal.shape}")
    p = []
    n = []
    for i in range(len(Y_label)):
        if Y_label[i]:
            p.append(hic_signal[i])
        else:
            n.append(hic_signal[i])
    print(f"Avg p {np.mean(p)} Avg n {np.mean(n)}")
    print(f"Max {np.max(hic_signal)} Min {np.min(hic_signal)}")
    inds = []
    for i, track in enumerate(head["expression"]):
        inds.append(i)
        # if "K562" in track:
        #     inds.append(i)
    print(f"K562 {len(inds)}")
    
    reducer = umap.UMAP()
    latent_vectors = reducer.fit_transform(dif)
    distances = np.asarray(distances)
    distances = np.expand_dims(distances, axis=1)
    add_features = np.concatenate((signals, signals_tss, hic_signal, distances), axis=-1) # ,fold_changes, cage_corr, atac_corr, ac_corr
    print(f"add_features shape is !!!!!!!!!! {add_features.shape}")
    print(dif.shape)
    dif = np.concatenate((dif, add_features), axis=-1)
    print(dif.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(dif, np.asarray(Y_label), test_size=0.3, random_state=1)
    print(f"X_train_1 shape is !!!!!!!!!! {X_train.shape}")
    X_train_dist = X_train[:, -1 * add_features.shape[1]:]
    X_test_dist = X_test[:, -1 * add_features.shape[1]:]
    X_train = X_train[:, :-1 * add_features.shape[1]]
    X_test = X_test[:, :-1 * add_features.shape[1]]
    print(f"X_train_dist shape is !!!!!!!!!! {X_train_dist.shape}")
    pos_num = 0
    neg_num = 0
    for label in Y_label:
        if label == 1:
            pos_num += 1
        if label == 0:
            neg_num += 1

    print(f"Pos {pos_num} Neg {neg_num}")
    print(f"X shape is !!!!!!!!!! {X_train.shape}")
    xd = np.copy(X_train)
    # transformer1 = RobustScaler().fit(xd)
    # xd = transformer1.transform(xd)
    pca = PCA(n_components=20)
    pca.fit(xd)
    # xd = pca.transform(xd)
    # transformer2 = RobustScaler().fit(xd)
    # vs just PCA?
    def transform(xx):
        # xx = transformer1.transform(xx)
        xx = pca.transform(xx)
        # xx = transformer2.transform(xx)
        return xx

    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train_dist, Y_train)
    Y_pred = clf.predict_proba(X_test_dist)[:,1]
    aucm = roc_auc_score(Y_test, Y_pred)
    print(f"AUCm: {aucm}")

    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict_proba(X_test)[:,1]
    auc0 = roc_auc_score(Y_test, Y_pred)
    print(f"AUC0: {auc0}")

    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(np.concatenate((X_train, X_train_dist), axis=-1), Y_train)
    Y_pred = clf.predict_proba(np.concatenate((X_test, X_test_dist), axis=-1))[:,1]
    auc1 = roc_auc_score(Y_test, Y_pred)
    print(f"AUC1: {auc1}")

    X_train = transform(X_train)
    X_test = transform(X_test)

    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict_proba(X_test)[:,1]
    auc2 = roc_auc_score(Y_test, Y_pred)
    print(f"AUC2: {auc2}")

    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(np.concatenate((X_train, X_train_dist), axis=-1), Y_train)
    Y_pred = clf.predict_proba(np.concatenate((X_test, X_test_dist), axis=-1))[:,1]
    auc3 = roc_auc_score(Y_test, Y_pred)
    print(f"AUC3: {auc3}") 

    fig, axs = plt.subplots(1,2,figsize=(20, 10))
    metrics.plot_roc_curve(clf, np.concatenate((X_test, X_test_dist), axis=-1), Y_test, ax=axs[0], name="Random Forest")
    axs[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # axs[0].xlim([0.0, 1.0])
    # axs[0].ylim([0.0, 1.05])
    # axs[0].xlabel("False Positive Rate")
    # axs[0].ylabel("True Positive Rate") 

    data = {'x': latent_vectors[:, 0],
            'y': latent_vectors[:, 1],
            'Type': region_type}
    df = pd.DataFrame(data)

    sns.scatterplot(x="x", y="y", hue="Type", data=df, s=50, alpha=0.5, ax=axs[1])
    axs[1].set_title("")
    axs[1].set_xlabel("UMAP1")
    axs[1].set_ylabel("UMAP2")

    plt.tight_layout()
    plt.savefig("linking.png")

    joblib.dump(clf, "RF.pkl") 
    joblib.dump(pca, "PCA.pkl")

    return auc3


if __name__ == '__main__':
    get_linking_AUC()