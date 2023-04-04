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
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from sklearn.linear_model import LogisticRegression
sns.set(font_scale = 1.5)


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
            if mark in track.lower() and "k562" in track.lower():
                chosen_tracks1.append(track)
        print(f"{len(chosen_tracks1)} {mark} tracks found")
        signal = parser.par_load_data(positions, chosen_tracks1, p)
        if do_mean:
            signal = np.mean(signal, axis=-1, keepdims=True)
        if signals is None:
            signals = signal
        else:
            signals = np.concatenate((signals, signal), axis=1)
        # print(f"{mark} shape is {signals.shape}")
    return signals


def get_seqs_and_features(df, one_hot, head):
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

    marks = ["h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3", "h3k36me3", "h3k27ac", "dnase"]
    signals = get_signals(mark_pos, head["epigenome"], marks)
    # signals_tss = get_signals(tss_pos, head["epigenome"], marks)

    hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    hic_signal = parser.par_load_hic_data_one(hic_keys, p, picked_regions_hic)
    hic_signal[np.isnan(hic_signal)] = 0
    hic_signal = np.expand_dims(hic_signal, axis=1)
    
    distances = np.expand_dims(np.asarray(distances), axis=1)
    # add_features = distances
    # Distances should be last in this array because they are used for binning. 
    add_features = np.concatenate((signals, hic_signal, distances), axis=-1)
    # print(f"Additional features shape is {add_features.shape}")
    return seqs1, seqs2, eseqs1, eseqs2, Y_label, add_features, tss_pos


def get_labels_and_scores(df, head):
    Y_label = []
    distances = []
    mark_pos = []
    picked_regions_hic = []
    tss_pos = []
    abc_scores = []
    for index, row in df.iterrows():   
        Y_label.append(row['Significant'])
        distances.append(abs(row["tss"] - row["mid"]))
        mark_pos.append((row["chr"], row["mid"] // p.bin_size))
        picked_regions_hic.append([row["chr"], min(row["mid"], row["tss"]), max(row["mid"], row["tss"])])
        tss_pos.append((row["chr"], row["tss"] // p.bin_size))
        abc_scores.append(row["ABC Score"])

    marks = ["h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3", "h3k36me3", "h3k27ac", "dnase"]
    signals = get_signals(mark_pos, head["epigenome"], marks)
    signals_tss = get_signals(tss_pos, head["epigenome"], marks)

    hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    hic_signal = parser.par_load_hic_data_one(hic_keys, p, picked_regions_hic)
    hic_signal[np.isnan(hic_signal)] = 0
    hic_signal = np.expand_dims(hic_signal, axis=1)
    
    abc_scores = np.asarray(abc_scores)
    abc_scores = np.expand_dims(np.asarray(abc_scores), axis=1)

    distances = np.expand_dims(np.asarray(distances), axis=1)
    abc_like_scores = get_signals(mark_pos, head["epigenome"], ["h3k27ac"]) / distances
    # add_features = distances
    # Distances should be last in this array because they are used for binning. 
    add_features = np.concatenate((signals, hic_signal, distances), axis=-1)
    return Y_label, add_features, abc_scores, abc_like_scores, tss_pos

def abc_vs_rf():
    head = joblib.load(f"{p.pickle_folder}heads.gz")["hg38"]
    df = pd.read_csv("data/enhancers/fulco2019_processed.tsv", sep="\t")
    Y_label, add_features, abc_scores, abc_like_scores, tss_pos = get_labels_and_scores(df, head)
    methods = ["ABC", "ABC*", "RF"]
    auc = {}
    for m in methods:
        auc[m] = 0
    indices = np.array([i for i in range(len(tss_pos))])
    unique_values = list(set(tss_pos))
    value_indices = {value: np.where([tss_pos[index] == value for index in indices])[0] for value in unique_values}
    K = 5
    kf = KFold(n_splits=K)
    fold1 = True
    for train_indices_unique, test_indices_unique in kf.split(unique_values):
        train_indices = np.concatenate([value_indices[value] for value in [unique_values[i] for i in train_indices_unique]])
        test_indices = np.concatenate([value_indices[value] for value in [unique_values[i] for i in test_indices_unique]])
        print(len(test_indices))
        for features in methods:
            print(features)
            if features == "RF":
                all_features = add_features
            elif features == "ABC":
                all_features = abc_scores
            elif features == "ABC*":
                all_features = abc_like_scores
            X_train = all_features[train_indices]
            Y_train = np.asarray(Y_label)[train_indices]
            X_test = all_features[test_indices]
            Y_test = np.asarray(Y_label)[test_indices]
            if features == "RF":
                clf = RandomForestClassifier(n_estimators=100) # n_estimators=500, max_depth=10
                clf.fit(X_train, Y_train)
                Y_pred = clf.predict_proba(X_test)[:,1]
                auc[features] += roc_auc_score(Y_test, Y_pred)
            else:
                clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
                Y_pred = clf.predict_proba(X_test)[:,1]
                auc[features] += roc_auc_score(Y_test, Y_pred)

            
    for features in methods:
        auc[features] = auc[features] / K
        print(f"{features}: {auc[features]}")


def get_linking_AUC():
    df = pd.read_csv("data/enhancers/all.tsv", sep="\t")
    print(f"Total rows: {len(df)}")
    print(f"Number of unique TSS: {df['tss'].nunique()}") 
    print(df['Significant'].value_counts())
    combination_counts = df.groupby(['tss', 'Significant']).size().reset_index(name='counts')
    print(combination_counts)
    print(df['Significant'].value_counts())
    print(df[df['Significant'] == True]['tss'].nunique())
    print(df[df['Significant'] == False]['tss'].nunique())
    head = joblib.load(f"{p.pickle_folder}heads.gz")["hg38"]
    for key in head.keys():
        print(f"Number of tracks in head {key}: {len(head[key])}")
    one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
    # df_tiling = pd.read_csv("data/enhancers/tiling_tss.tsv", sep="\t")
    seqs1, seqs2, eseqs1, eseqs2, Y_label, add_features, tss_pos = get_seqs_and_features(df, one_hot, head)

    true_labels = {"<20000":[], "<40000":[], ">40000":[]}
    methods = ["ABC like", "Enformer", "Chromix",  "Enformer*", "Chromix*", "ChromixF"] 
    pred_labels = {}
    for m in methods:
        pred_labels[m] = {"<20000":[], "<40000":[], ">40000":[]}
    auc = {}
    for m in methods:
        auc[m] = 0
    print(f"Predicting effect of {len(seqs1)} sequences")

    import enformer_usage
    enformer_effect = enformer_usage.calculate_effect(np.asarray(eseqs1), np.asarray(eseqs2))
    joblib.dump(enformer_effect, "enformer_effect_sum.p", compress=3)
    # enformer_effect = joblib.load("enformer_effect_sum.p")

    import tensorflow as tf
    import model as mo
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 6, p.hic_size, head)
        our_model.get_layer("our_stem").set_weights(joblib.load(p.model_path + "_stem"))
        our_model.get_layer("our_body").set_weights(joblib.load(p.model_path + "_body"))
        our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
        our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
        our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
        our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

        # stem = our_model.get_layer("our_stem")
        # body = our_model.get_layer("our_body")
        # submodel = Model(stem.input, body(stem.output))
    # chomix_effects_e = mo.batch_predict_effect_x(p, submodel, np.asarray(seqs1), np.asarray(seqs2))
    # joblib.dump(chomix_effects_e, "chomix_effects_e.p", compress=3)
    # chomix_effects_e = joblib.load("chomix_effects_e.p")

    chomix_effects_e, chromix_effects_h, chromix_fold_changes = mo.batch_predict_effect(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
    joblib.dump((chomix_effects_e, chromix_effects_h, chromix_fold_changes), "chromix_full_effect_ce.p", compress=3)
    # chomix_effects_e, chromix_effects_h, chromix_fold_changes = joblib.load("chromix_full_effect_ce.p")

    # for features in methods:
    #     print(features)
    #     yinds = np.asarray(Y_label).argsort()
    #     sorted_dif = chomix_effects_e[yinds[::-1]]
    #     # sorted_dif = np.log10(sorted_dif + 1)
    #     sorted_y = np.asarray(Y_label)[yinds[::-1]]
    #     palette = sns.color_palette()
    #     palette_dict = dict(zip(["Significant", "Non-Significant"], palette))
    #     pair_labels = ["Significant", "Non-Significant"]
    #     pair_colors = []
    #     for y in sorted_y:
    #         if y == True:
    #             pair_colors.append(palette_dict["Significant"])
    #         else:
    #             pair_colors.append(palette_dict["Non-Significant"])
    #     cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
    #     g = sns.clustermap(sorted_dif,row_cluster=False, col_cluster=False,row_colors=pair_colors,
    #                       linewidths=0, xticklabels=False, yticklabels=False)
    #     for label in pair_labels:
    #         g.ax_col_dendrogram.bar(0, 0, color=palette_dict[label],
    #                                 label=label, linewidth=0)
    #     g.ax_col_dendrogram.legend(loc="center", ncol=5)
    #     g.cax.set_position([.97, .2, .03, .45])
    #     g.savefig(f"{features}_heatmap.png")

    indices = np.array([i for i in range(len(tss_pos))])
    unique_values = list(set(tss_pos))
    value_indices = {value: np.where([tss_pos[index] == value for index in indices])[0] for value in unique_values}
    K = 5
    kf = KFold(n_splits=K)
    fold1 = True
    for train_indices_unique, test_indices_unique in kf.split(unique_values):
        train_indices = np.concatenate([value_indices[value] for value in [unique_values[i] for i in train_indices_unique]])
        test_indices = np.concatenate([value_indices[value] for value in [unique_values[i] for i in test_indices_unique]])
        for features in methods:
            print(features)
            # mid_bin = dif.shape[1] // 2
            # mid_val = dif[:, mid_bin - 1] + dif[:, mid_bin] + dif[:, mid_bin + 1]
            # dif = np.concatenate((np.expand_dims(mid_val, axis=1), add_features), axis=-1)
            if features == "ABC like":
                all_features = add_features
            elif features.startswith("Chromix"):
                all_features = chomix_effects_e
            elif features == "Enformer":
                all_features = enformer_effect
            X_train = all_features[train_indices]
            Y_train = np.asarray(Y_label)[train_indices]
            X_test = all_features[test_indices]
            Y_test = np.asarray(Y_label)[test_indices]

            if features != "ABC like":
                pca = PCA(n_components=20)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

                if features in ["ChromixF"]:
                    X_train_h = chromix_effects_h[train_indices, :]
                    X_test_h = chromix_effects_h[test_indices, :]
                    X_train_h = np.mean(X_train_h, axis=-1, keepdims=True)
                    X_test_h = np.mean(X_test_h, axis=-1, keepdims=True)
                    X_train = np.concatenate((X_train, X_train_h), axis=-1)
                    X_test = np.concatenate((X_test, X_test_h), axis=-1)

            X_train_dist = add_features[train_indices]
            X_test_dist = add_features[test_indices]
            if features != "ABC like":
                if "*" in features or "3D" in features:
                    X_test = np.concatenate((X_test, X_test_dist), axis=-1)
                    X_train = np.concatenate((X_train, X_train_dist), axis=-1)
                else:
                    X_test = np.concatenate((X_test, X_test_dist[:, -1:]), axis=-1)
                    X_train = np.concatenate((X_train, X_train_dist[:, -1:]), axis=-1)

            clf = RandomForestClassifier(n_estimators=100) # n_estimators=500, max_depth=10
            print(f"Fitting {X_train.shape}")
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict_proba(X_test)[:,1]
            auc[features] += roc_auc_score(Y_test, Y_pred)
            if fold1:
                scores = {True:[], False:[]}
                for i in range(len(Y_test)):
                    scores[Y_test[i]].append(clf.predict_proba(np.asarray([X_test[i]]))[0,1])

                fig, axs = plt.subplots(1, 1, figsize=(15, 10))
                # Plot the distributions using seaborn's distplot
                sns.distplot(scores[True], hist=True, label='True (count: {})'.format(len(scores[True])))
                sns.distplot(scores[False], hist=True, label='False (count: {})'.format(len(scores[False])))
                plt.legend()
                # Add labels and title
                plt.xlabel('Score')
                plt.ylabel('Density')
                plt.title('Distributions of Scores for True and False')
                plt.savefig(f"Distributions_fold1.png")
            # if fold1:
            #     # Saving RF and PCA for other scripts
            #     if features == "Chromix":
            #         joblib.dump(clf, "RF.pkl")
            #         joblib.dump(pca, "PCA.pkl")
            #     # Saving ROC curve
            #     plt.clf()
            #     fig, axs = plt.subplots(1,1,figsize=(10, 10))
            #     metrics.plot_roc_curve(clf, X_test, Y_test, ax=axs, name="Random Forest")
            #     axs.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            #     plt.tight_layout()
            #     plt.savefig(features + "_linking.png")
            # Y_pred = clf.predict(X_test)
            # # Distance for grouping results
            # X_test_dist = X_test_dist[:, -1:]
            # for i in range(len(Y_pred)):
            #     if X_test_dist[i] < 20000:
            #         dist_group = "<20000"
            #     elif X_test_dist[i] < 40000:
            #         dist_group = "<40000"
            #     else:
            #         dist_group = ">40000"
            #     if features == methods[0]:
            #         true_labels[dist_group].append(Y_test[i])
            #     pred_labels[features][dist_group].append(Y_pred[i])
        fold1=False
    # for metric in ["Recall", "Precision"]:
    #     df = pd.DataFrame(columns=['Distance', 'Method', metric])
    #     for key in true_labels.keys():
    #         for m in methods:
    #             cf = confusion_matrix(true_labels[key], pred_labels[m][key])
    #             tn = cf[0][0]
    #             fp = cf[0][1]
    #             fn = cf[1][0]
    #             tp = cf[1][1]
    #             if metric == "Recall":
    #                 v = tp/(tp+fn)
    #             else:
    #                 v = tp/(tp+fp)
    #             df.loc[len(df.index)] = [f"{key} ({(tp+fn) / K})", m, v]
    #     plt.clf()
    #     fig, axs = plt.subplots(1,1,figsize=(20, 10))
    #     sns.barplot(x="Distance", hue="Method", y=metric, data=df, palette="Set2", ax=axs)
    #     plt.tight_layout()
    #     plt.savefig(f"linking_comparison_{metric}.png")
    for features in methods:
        auc[features] = auc[features] / K
        print(f"{features}: {auc[features]}")


    plt.close(fig)
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(auc, orient='index', columns=['Values'])

    # Create the bar plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    sns.barplot(x=df.index, y='Values', data=df, ax=axs)
    axs.set_ylim(0.75, 0.85)

    # Add labels and title
    plt.xlabel('Methods')
    plt.ylabel('AUC')
    # plt.title('Values by Names')

    plt.tight_layout()
    plt.savefig(f"barplot1.png")

    return auc["Chromix"]


def linking_proba(chrom, tss, enhancer_mids, one_hot, our_model):
    import model as mo
    df = pd.DataFrame(list(zip([chrom]*len(tss), tss, enhancer_mids, [False]*len(tss))),
                    columns =['chr', 'tss', 'mid', 'Significant'])
    seqs1, seqs2, eseqs1, eseqs2, Y_label, add_features = get_seqs_and_features(df, one_hot)
    clf = joblib.load("RF.pkl") 
    # pca = joblib.load("PCA.pkl")
    if len(seqs1) > 0:
        dif, fold_changes = mo.batch_predict_effect_x(p, our_model, np.asarray(seqs1), np.asarray(seqs2))
        # dif = pca.transform(dif)
        pred = clf.predict_proba(np.concatenate((dif, add_features), axis=-1))[:,1]
        with open(f"snp_linking/{chrom}_{enhancer_mids[0]}_{max(pred)}.bedGraph", 'w+') as file:
            for i in range(len(pred)):
                file.write(f"{chrom}\t{tss[i]}\t{tss[i]}\t{pred[i]}\n")
    else:
        pred = [0]
    return pred

p = MainParams()
if __name__ == '__main__':
    # abc_vs_rf()
    get_linking_AUC()