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
sns.set(font_scale = 2.5)
import model as mo
from main_params import MainParams
import common as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import umap
from sklearn.metrics import RocCurveDisplay


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


p = MainParams()

head = joblib.load(f"{p.pickle_folder}heads.gz")

one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")


model_path = p.model_folder + "0.8070915954746051_0.5100707215128535/" + p.model_name 
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head)
    our_model.get_layer("our_resnet").set_weights(joblib.load(model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(model_path + "_expression"))


df = pd.read_csv("data/fulco.tsv", sep="\t")
df["mid"] = df["start"] + (df["end"] - df["start"]) // 2
df_pos = df.loc[df['Significant'] == True]
df_neg = df.nlargest(len(df_pos),'Adjusted p-value')
df = pd.concat([df_pos, df_neg], ignore_index=True, axis=0)

seqs1 = []
seqs2 = []
Y_label = []
for index, row in df.iterrows():
    start = row["Gene TSS"] - p.half_size - 1
    if row["chr"] not in one_hot.keys():
        continue

    if start < 0 or start + p.input_size > len(one_hot[row["chr"]]):
        continue

    Y_label.append(row["Significant"])
    relative = row["Gene TSS"] - row["mid"]
    enh_pos = p.half_size - relative 
    seq1 = one_hot[row["chr"]][start: start + p.input_size][..., :-1]
    seqs1.append(seq1)
    seq2 = seq1.copy()
    seq2[enh_pos - 1000: enh_pos + 1000] = np.zeros(seq2[enh_pos - 1000: enh_pos + 1000].shape)
    # np.random.permutation(seq2[enh_pos - 1000: enh_pos + 1000])
    seqs2.append(seq2)
    # if len(seqs1) > 500:
    #     break
fig, axs = plt.subplots(1,1,figsize=(10, 10))
for t in range(2):
    inds = []
    for i, track in enumerate(head["expression"]):
        if t==0:
            inds.append(i)
        elif "K562" in track:
            inds.append(i)
    print(f"K562 {len(inds)}")

    print(f"Predicting {len(seqs1)}")
    dif = mo.batch_predict_effect2(p, our_model, np.asarray(seqs1), np.asarray(seqs2), inds)
    print("Done")
    joblib.dump(dif, "fulco_dif.p" + str(t), compress=3)
    print("Dumped")
    # dif = joblib.load("fulco_dif.p")

    reducer = umap.UMAP()
    latent_vectors = reducer.fit_transform(dif)
    
    print("Plotting")
    data = {'x': latent_vectors[:, 0],
            'y': latent_vectors[:, 1],
            'Enhancer': Y_label}

    df = pd.DataFrame(data)

    # sns.scatterplot(x="x", y="y", hue="Enhancer", data=df, s=50, alpha=0.5, ax=axs)
    # axs.set_title("")
    # axs.set_xlabel("UMAP1")
    # axs.set_ylabel("UMAP2")
    # plt.tight_layout()
    # plt.savefig("fulco.png")



    X_train, X_test, Y_train, Y_test = train_test_split(dif, np.asarray(Y_label), test_size=0.3) # , random_state=1

    pos_num = 0
    neg_num = 0
    for label in Y_label:
        if label == 1:
            pos_num += 1
        if label == 0:
            neg_num += 1

    print(f"Pos {pos_num} Neg {neg_num}")

    xd = np.copy(X_train)
    transformer1 = RobustScaler().fit(xd)
    xd = transformer1.transform(xd)
    pca = PCA(n_components=50)
    pca.fit(xd)
    xd = pca.transform(xd)
    transformer2 = RobustScaler().fit(xd)

    def transform(xx):
        xx = transformer1.transform(xx)
        xx = pca.transform(xx)
        xx = transformer2.transform(xx)
        return xx


    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
    auc1 = metrics.auc(fpr, tpr)
    print(f"AUC1: {auc1}")

    X_train = transform(X_train)
    X_test = transform(X_test)
    clf = RandomForestClassifier() # , max_depth=20
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
    auc2 = metrics.auc(fpr, tpr)
    print(f"AUC2: {auc2}")

    if t==0:
        name = "ALL"
    else:
        name = "K562"
    metrics.plot_roc_curve(clf, X_test, Y_test, ax=axs, name=name)

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate") 
plt.tight_layout()
plt.savefig("fulco_roc.png")