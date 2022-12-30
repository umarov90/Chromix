import os
import pathlib
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
import parse_data as parser
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

MAX_TRACKS = 1
MAX_PROMOTERS = 5
k = 40
OUT_DIR = "motifs/"


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


p = MainParams()
head = joblib.load(f"{p.pickle_folder}heads.gz")
head_tracks = head["expression"]
one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
heads = joblib.load(f"{p.pickle_folder}heads.gz")
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head_tracks)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))


train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
eval_infos = valid_info + test_info
test_seq = []
for info in eval_infos:
    start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
    extra = start + p.input_size - len(one_hot[info[0]])
    if start < 0:
        ns = one_hot[info[0]][0:start + p.input_size]
        ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
    elif extra > 0:
        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
        ns = np.concatenate((ns, np.zeros((extra, 5))))
    else:
        ns = one_hot[info[0]][start:start + p.input_size]
    test_seq.append(ns[:, :-1])
test_seq = np.asarray(test_seq, dtype=bool)

tracks_count = 0
for track_to_use, track in enumerate(head_tracks):
    if "FANTOM5" not in track or "K562" not in track or "response" in track:
        continue
    seqs_to_explain = []
    shap_values = []
    print(track)
    for si, seq in enumerate(test_seq):
        if si % 1 == 0:
            print(si, end=" ")
        if si > MAX_PROMOTERS:
            break
        seqs_to_explain.append(seq)
        # attribution
        baseline = tf.zeros(shape=(p.input_size, p.num_features))
        image = seq.astype('float32')
        ig_attributions = attribution.integrated_gradients(our_model, baseline=baseline,
                                                           image=image,
                                                           target_class_idx=[p.mid_bin, track_to_use],
                                                           m_steps=10)

        attribution_mask = tf.squeeze(ig_attributions).numpy()
        shap_values.append(attribution_mask)

    np.savez_compressed("ohe.npz", np.asarray(seqs_to_explain))
    np.savez_compressed("shap.npz", np.asarray(seqs_to_explain))


