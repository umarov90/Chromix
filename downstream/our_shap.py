import shap
import numpy as np
import os
import pathlib
from skimage import measure
import model as mo
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import attribution
import logomaker
from sklearn.cluster import KMeans
from main_params import MainParams
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

MAX_TRACKS = 2
MAX_PROMOTERS = 2
OUT_DIR = "temp/"


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
test_info = joblib.load("pickle/test_info.gz")
gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
tss_loc = joblib.load("pickle/tss_loc.gz")
for key in tss_loc.keys():
    tss_loc[key].sort()

# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# with strategy.scope():
our_model = tf.keras.models.load_model(model_folder + p.model_name)
our_model.get_layer("our_head").set_weights(joblib.load(model_folder + p.model_name + "_head_" + str(head_id)))

print("Loading")
test_seq = joblib.load(f"pickle/chr1_seq.gz")
train_seq = joblib.load(f"pickle/chr2_seq.gz")
print("SHAP")
# select a set of background examples to take an expectation over
background = train_seq[np.random.choice(train_seq.shape[0], 2, replace=False)]

# explain predictions of the model on three images
e = shap.DeepExplainer((our_model.input, our_model.output[:, :, p.mid_bin]), background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(test_seq[:5])
print(len(shap_values))

