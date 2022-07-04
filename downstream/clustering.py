import tensorflow as tf
import joblib
from main_params import MainParams
import visualization as viz
import model as mo
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import parse_data as parser
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
matplotlib.use("Agg")

p = MainParams()
head_name = "hg38"
heads = joblib.load(f"{p.pickle_folder}heads.gz")
head = heads[head_name]["expression"]
cor_tracks = pd.read_csv("data/fantom_tracks.tsv", sep="\t", header=None).iloc[:, 0].tolist()
cor_inds = {}
short_name = {}
colors = []
inds = []
for i, t in enumerate(head):
    if "FANTOM5" not in t:
        continue
    track_type = t[:t.index(".")]
    colors.append(track_type)
    start = t.index(".ctss.") + len(".ctss.")
    end = t.find("_") # t.find("_", t.find("_") + 1)
    if end == -1:
        end = t.find(".", start)
    inds.append(i)
    if t in cor_tracks:
        cor_inds[t] = len(inds) - 1
        short_name[t] = t[start:end]
weights = joblib.load(p.model_path + "_expression_hg38")
weights = np.vstack([np.squeeze(weights[0]), weights[1]]).T
weights = weights[inds, :]
reducer = umap.UMAP()
latent_vectors = reducer.fit_transform(weights)
fig, axs = plt.subplots(1,1,figsize=(10, 10))
print("Plotting")
data = {'x': latent_vectors[:, 0],
        'y': latent_vectors[:, 1],
        'c': colors}

df = pd.DataFrame(data)

sns.scatterplot(x="x", y="y", hue="c", data=df, s=5, alpha=0.2, ax=axs)
for i, track in enumerate(cor_tracks):
    axs.text(latent_vectors[cor_inds[track], 0], latent_vectors[cor_inds[track], 1], short_name[track], color="black", fontsize=6)
axs.set_title("Latent space")
axs.set_xlabel("A1")
axs.set_ylabel("A2")
plt.tight_layout()
plt.savefig("clustering.svg")