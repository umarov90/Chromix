# import model as mo
# import tensorflow as tf
from main_params import MainParams
import joblib
import numpy as np
import pandas as pd
import parse_data as parser
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import RocCurveDisplay
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
import umap
import seaborn as sns
sns.set(font_scale = 2.5)

p = MainParams()


one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")
step = 640
output_scores_info = []
head = joblib.load(f"{p.pickle_folder}heads.gz")

inds = []
for i, track in enumerate(head["expression"]):
    inds.append(i)

# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# with strategy.scope():
#     our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head["expression"])
#     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))

effect = None
for chrom in ["chr11"]: # one_hot.keys()
    seqs1 = []
    seqs2 = []
    print(f"\nPredicting {chrom} +++++++++++++++++++++++")
    for expression_region in range(0, len(one_hot[chrom]), step):
        start = expression_region - p.half_size
        extra = start + p.input_size - len(one_hot[chrom])
        if start < 0 or extra > 0:
            continue # dont want special cluster
        mid = expression_region // p.bin_size
        output_scores_info.append([chrom, mid])
#         ns = one_hot[chrom][start:start + p.input_size]
#         seq1 = ns[..., :-1]
#         seqs1.append(seq1)
#         seq2 = seq1.copy()
#         seq2[p.half_size - step // 2: p.half_size + step // 2] = 0
#         seqs2.append(seq2)
#         if len(seqs1) % 1000 == 0:
#             print(f"[{expression_region}]", end=" ")
#             new_effect = mo.batch_predict_effect2(p, our_model, np.asarray(seqs1), np.asarray(seqs2), inds)
#             if effect is None:
#                 effect = new_effect
#             else:
#                 effect = np.concatenate((effect, new_effect))
#             seqs1 = []
#             seqs2 = []
#             # if len(effect) > 20000:
#             #     break
#     if len(seqs1) > 0:
#         new_effect = mo.batch_predict_effect2(p, our_model, np.asarray(seqs1), np.asarray(seqs2), inds)
#         effect = np.concatenate((effect, new_effect))

# joblib.dump(effect, "ism_region_effect.p", compress=3)
# print("Dumped")
effect = joblib.load("ism_region_effect.p")
print(f"Loaded effects {len(effect)}")
head1 = head["epigenome"]

pca = PCA(n_components=20)
pca.fit(effect)
pca_effect = pca.transform(effect)
reducer = umap.UMAP()
latent_vectors = reducer.fit_transform(pca_effect)

print("Plotting")
data = {'x': latent_vectors[:, 0],
        'y': latent_vectors[:, 1]}  # , 'z': output_epigenome1
df = pd.DataFrame(data)

colors = ['green','orange','brown','dodgerblue','red', 'purple']
marks = ["h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3", "h3k36me3", "h3k27ac"]

fig, axs = plt.subplots(2, len(marks) // 2, figsize=(len(marks) * 5, 15))
axs = axs.flatten()
# pal = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
# sns.scatterplot(x="x", y="y",  data=df, s=10, alpha=0.05, ax=axs) # hue="z", palette=pal


for i, mark in enumerate(marks):
    chosen_tracks1 = []
    for track in head1:
        if mark in track.lower():
            chosen_tracks1.append(track)
    print(f"{len(chosen_tracks1)} {mark} tracks found")
    output_epigenome1 = parser.par_load_data(output_scores_info, chosen_tracks1, p)
    print(output_epigenome1.shape)
    output_epigenome1 = np.mean(output_epigenome1, axis=1)#[:len(effect)]
    output_epigenome1 = 0.05 * (output_epigenome1 / np.max(output_epigenome1))
    print(output_epigenome1.shape)

    r, g, b = to_rgb(colors[i])
    color = [(r, g, b, alpha) for alpha in output_epigenome1]
    axs[i].scatter(latent_vectors[:, 0], latent_vectors[:, 1], c="gray", s=10)
    axs[i].scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=color, s=10)
    axs[i].set_title(mark)
    axs[i].set_xlabel("UMAP1")
    axs[i].set_ylabel("UMAP2")

# custom_lines = []
# for c in colors:
#     custom_lines.append(Line2D([0], [0], color=c, marker='o', markerfacecolor=c, markersize=15))
# axs.legend(custom_lines, marks)


plt.tight_layout()
plt.savefig("ism_regions_umap.png")
print("Saved")