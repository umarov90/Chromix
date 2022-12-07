import os
import pandas as pd
import pathlib
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
from scipy import stats
import bisect 
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(font_scale = 2.5)


def take_closest(myList, myNumber):
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


def add_seqs(chrom, tss_pos, enh_mid, input_size, one_hot):
    half_size = input_size // 2 
    start = tss_pos - half_size - 1
    if start < 0 or start + input_size > len(one_hot[chrom]):
        # print("edge")
        return None, None, None
    relative1 = tss_pos - enh_mid
    if abs(relative1) > half_size or abs(relative1) < 2000: 
        # print(f"bad {relative1}")
        return None, None, None
    enh_pos = half_size - relative1
    fseq1 = one_hot[chrom][start: start + input_size][..., :-1]     
    fseq2 = fseq1.copy()
    fseq2[enh_pos - 1000: enh_pos + 1000] = 0
    return fseq1, fseq2, relative1


def get_signals(positions, head, kw, do_mean=True):
    signals = None
    for i, mark in enumerate(kw):
        chosen_tracks1 = []
        for track in head:
            if mark in track.lower():
                chosen_tracks1.append(track)
        print(f"{len(chosen_tracks1)} {mark} tracks found")
        signal = parser.par_load_data_temp(positions, chosen_tracks1, p)
        if do_mean:
            signal = np.mean(signal, axis=-1, keepdims=True)
        if signals is None:
            signals = signal
        else:
            signals = np.concatenate((signals, signal), axis=1)
    print(f"{mark} shape is {signals.shape}")
    return signals


def get_seqs_and_features(df, one_hot):
    seqs1 = []
    seqs2 = []
    eseqs1 = []
    eseqs2 = []
    Y_label = []
    distances = []
    mark_pos = []
    picked_regions_hic = []
    tss_pos = []
    picked_pairs = []
    for index, row in df.iterrows():    
        # if row["chr"] not in one_hot.keys():
        #     continue
        pair_id = row["chr"] + ":" + str(row["tss"] // p.bin_size) + ":" + str(row["mid"] // p.bin_size)
        if pair_id in picked_pairs:
            # print("Skipping " + pair_id)
            continue
        seq1, seq2, relative = add_seqs(row["chr"], row["tss"], row["mid"], p.input_size, one_hot)
        eseq1, eseq2, erelative = add_seqs(row["chr"], row["tss"], row["mid"], 393216, one_hot)
        if relative is not None and erelative is not None: 
            picked_pairs.append(pair_id)
            Y_label.append(row['Significant'])
            distances.append(abs(relative))
            mark_pos.append((row["chr"], row["mid"] // p.bin_size))
            picked_regions_hic.append([row["chr"], min(row["mid"], row["tss"]), max(row["mid"], row["tss"])])
            tss_pos.append((row["chr"], row["tss"] // p.bin_size))
            seqs1.append(seq1)
            seqs2.append(seq2)
            eseqs1.append(eseq1)
            eseqs2.append(eseq2)

    # print(f"Picked pairs: {len(picked_pairs)}")
    # print(f"TSS number: {len(tss_pos)} {len(set(tss_pos))}")
    # print(f"Putative enhancer number: {len(mark_pos)} {len(set(mark_pos))}")
    # print(f"Significant: {Y_label.count(True)}, Non-Significant: {Y_label.count(False)}")
    # print(np.quantile(np.asarray(distances), [0.33, 0.66, 0.99]))
    # np.savetxt("distances.csv", np.asarray(distances), delimiter=",")

    # marks = ["h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3", "h3k36me3", "h3k27ac"]
    # signals = get_signals(mark_pos, head["epigenome"], marks)
    # signals_tss = get_signals(tss_pos, head["epigenome"], marks)

    # hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    # hic_signal = parser.par_load_hic_data_one(hic_keys, p, picked_regions_hic)
    # hic_signal[np.isnan(hic_signal)] = 0
    # hic_signal = np.expand_dims(hic_signal, axis=1)
    
    distances = np.expand_dims(np.asarray(distances), axis=1)
    add_features = distances
    # add_features = np.concatenate((signals, signals_tss, hic_signal, distances), axis=-1)
    # print(f"Additional features shape is {add_features.shape}")
    return seqs1, seqs2, eseqs1, eseqs2, Y_label, add_features


def get_linking_AUC():
    head = joblib.load(f"{p.pickle_folder}heads.gz")
    one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")
    df = pd.read_csv("data/enhancers/all.tsv", sep="\t")
    # df_tiling = pd.read_csv("data/enhancers/tiling_tss.tsv", sep="\t")
    seqs1, seqs2, eseqs1, eseqs2, Y_label, add_features = get_seqs_and_features(df, one_hot)

    true_labels = {"<40000":[], "<100000":[], ">100000":[]}
    methods = ["Enformer", "Chromix"] # "Baseline", 
    pred_labels = {}
    for m in methods:
        pred_labels[m] = {"<40000":[], "<100000":[], ">100000":[]}
    auc = {}
    for features in methods:
        if features == "Chromix": 
            import tensorflow as tf
            import model as mo
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            with strategy.scope():
                our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head["expression"])
                our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
                our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
            # dif, fold_changes = mo.batch_predict_effect_x(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
            dif = mo.batch_predict_effect2(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
            joblib.dump(dif, "chromix_effect.p", compress=3)
            # dif = joblib.load("chromix_effect.p")
        elif features == "Enformer":
            import enformer_usage
            dif = enformer_usage.calculate_effect(np.asarray(eseqs1), np.asarray(eseqs2))
            joblib.dump(dif, "enformer_effect.p", compress=3)
            # dif = joblib.load("enformer_effect.p")
        yinds = np.asarray(Y_label).argsort()
        sorted_dif = dif[yinds[::-1]]
        fig, axs = plt.subplots(1, 1, figsize=(dif.shape[1] // 100, dif.shape[0] // 100))
        sns.heatmap(sorted_dif, ax=axs)
        plt.tight_layout()
        plt.savefig(f"{features}_heatmap.png")
        if features == "Baseline":
            dif = add_features
        else:
            dif = np.concatenate((dif, add_features), axis=-1)
        print(f"Y_label shape is {np.asarray(Y_label).shape}")
        X_train, X_test, Y_train, Y_test = train_test_split(dif, np.asarray(Y_label), test_size=0.3, random_state=0)
        X_train_dist = X_train[:, -1 * add_features.shape[1]:]
        X_test_dist = X_test[:, -1 * add_features.shape[1]:]
        X_train = X_train[:, :-1 * add_features.shape[1]]
        X_test = X_test[:, :-1 * add_features.shape[1]]

        if features != "Baseline":
            pca = PCA(n_components=10)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        X_test = np.concatenate((X_test, X_test_dist), axis=-1)
        X_train = np.concatenate((X_train, X_train_dist), axis=-1)

        clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict_proba(X_test)[:,1]
        auc[features] = roc_auc_score(Y_test, Y_pred)
        print(f"AUC: {auc[features]}")

        # Saving ROC curve
        plt.clf()
        fig, axs = plt.subplots(1,1,figsize=(10, 10))
        metrics.plot_roc_curve(clf, X_test, Y_test, ax=axs, name="Random Forest")
        axs.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.tight_layout()
        plt.savefig(features + "_linking.png")

        # Saving RF and PCA for other scripts
        if features == "Chromix":
            joblib.dump(clf, "RF.pkl") 
            joblib.dump(pca, "PCA.pkl")

        Y_pred = clf.predict(X_test)
        # Distance for grouping results
        X_train_dist = X_train_dist[:, -1:]
        for i in range(len(Y_pred)):
            if X_train_dist[i] < 40000:
                dist_group = "<40000"
            elif X_train_dist[i] < 100000:
                dist_group = "<100000"
            else:
                dist_group = ">100000"
            if features == methods[0]:
                true_labels[dist_group].append(Y_test[i])
            pred_labels[features][dist_group].append(Y_pred[i])

    for metric in ["Recall", "Precision"]:
        df = pd.DataFrame(columns=['Distance', 'Method', metric])
        for key in true_labels.keys():
            for m in methods:
                cf = confusion_matrix(true_labels[key], pred_labels[m][key])
                tn = cf[0][0]
                fp = cf[0][1]
                fn = cf[1][0]
                tp = cf[1][1]
                if metric == "Recall":
                    v = tp/(tp+fn)
                else:
                    v = tp/(tp+fp)
                df.loc[len(df.index)] = [f"{key} ({tp+fn})", m, v]
        plt.clf()
        fig, axs = plt.subplots(1,1,figsize=(20, 10))
        sns.barplot(x="Distance", hue="Method", y=metric, data=df, palette="Set2", ax=axs)
        plt.tight_layout()
        plt.savefig(f"linking_comparison_{metric}.png")
    return auc["Chromix"]


def linking_proba(chrom, tss, enhancer_mids, one_hot, our_model):
    import model as mo
    df = pd.DataFrame(list(zip([chrom]*len(tss), tss, enhancer_mids, [False]*len(tss))),
                    columns =['chr', 'tss', 'mid', 'Significant'])
    seqs1, seqs2, eseqs1, eseqs2, Y_label, add_features = get_seqs_and_features(df, one_hot)
    clf = joblib.load("RF.pkl") 
    pca = joblib.load("PCA.pkl")
    if len(seqs1) > 0:
        dif, fold_changes = mo.batch_predict_effect_x(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
        dif = pca.transform(dif)
        pred = clf.predict_proba(np.concatenate((dif, add_features), axis=-1))[:,1]
        with open(f"snp_linking/{chrom}_{enhancer_mids[0]}_{max(pred)}.bedGraph", 'w+') as file:
            for i in range(len(pred)):
                file.write(f"{chrom}\t{tss[i]}\t{tss[i]}\t{pred[i]}\n")
    else:
        pred = [0]
    return pred

p = MainParams()
if __name__ == '__main__':
    get_linking_AUC()