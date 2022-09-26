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
MAX_PROMOTERS = 50
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

model_path = p.model_folder + "0.8070915954746051_0.5100707215128535/" + p.model_name 

our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head_tracks)
our_model.get_layer("our_resnet").set_weights(joblib.load(model_path + "_res"))
our_model.get_layer("our_expression").set_weights(joblib.load(model_path + "_expression"))


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

meme_motifs = []
tracks_count = 0
for track_to_use, track in enumerate(head_tracks):
    if "FANTOM5" not in track or "K562" not in track or "response" in track:
        continue
    print(track)
    picked_sub_seqs = []
    picked_sub_seqs_scores = []
    for si, seq in enumerate(test_seq):
        if si % 1 == 0:
            print(si, end=" ")
        if si > MAX_PROMOTERS:
            break
        # attribution
        baseline = tf.zeros(shape=(p.input_size, p.num_features))
        image = seq.astype('float32')
        ig_attributions = attribution.integrated_gradients(our_model, baseline=baseline,
                                                           image=image,
                                                           target_class_idx=[p.mid_bin, track_to_use],
                                                           m_steps=10)

        attribution_mask = tf.squeeze(ig_attributions).numpy()
        attribution_mask = (attribution_mask - np.min(attribution_mask)) / (
                np.max(attribution_mask) - np.min(attribution_mask))
        attribution_mask = np.sum(attribution_mask[:, :4], axis=-1)
        start = int(p.input_size / 2) - 1500
        attribution_mask = attribution_mask[start: start + 3000]
        top_n = 100
        top = attribution_mask.argsort()[-top_n:][::-1]
        picked = []
        for v in top:
            if len(picked) > 0:
                nv = find_nearest(picked, v)
                if abs(v - nv) < 10:
                    continue
            picked_sub_seqs.append(seq[start + v - 5: start + v + 5][:, :4].flatten())
            picked_sub_seqs_scores.append(attribution_mask[v])
            picked.append(v)
            if len(picked) >= 40:
                break
    picked_sub_seqs = np.asarray(picked_sub_seqs)
    cluster_scores = {}
    kmeans = KMeans(n_clusters=min(k, len(picked_sub_seqs)), random_state=0).fit(picked_sub_seqs)
    for i in range(len(picked_sub_seqs)):
        cluster_scores.setdefault(kmeans.labels_[i], []).append(picked_sub_seqs_scores[i])
    for ci, cluster in enumerate(kmeans.cluster_centers_):
        cluster = cluster.reshape((-1, 4))
        fig, ax = plt.subplots(figsize=(16, 6))
        cluster_df = pd.DataFrame({'A': cluster[:, 0], 'C': cluster[:, 1], 'G': cluster[:, 2], 'T': cluster[:, 3]})
        # create Logo object
        crp_logo = logomaker.Logo(cluster_df,
                                  shade_below=.5,
                                  fade_below=.5,
                                  font_name='Arial Rounded MT Bold')

        # style using Logo methods
        crp_logo.style_spines(visible=False)
        crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
        crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

        # style using Axes methods
        crp_logo.ax.set_ylabel("Y", labelpad=-1)
        crp_logo.ax.xaxis.set_ticks_position('none')
        crp_logo.ax.xaxis.set_tick_params(pad=-1)
        plt.savefig(f"{OUT_DIR}{track}_logo{ci + 1}_{np.mean(cluster_scores[ci])}.png")
        plt.close(fig)
        meme_motifs.append(f"MOTIF {ci + 1}\nletter-probability matrix:"
                           f" alength= 4 w= 10 nsites= {len(cluster_scores[ci])}"
                           f" E= {np.mean(cluster_scores[ci])}\n{cluster_df.to_string(header=False, index=False)}")

    header = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"

    with open(f"{OUT_DIR}{track}_motifs.meme", "w") as f:
        f.write(header)
        for s in meme_motifs:
            f.write(s)
            f.write("\n\n")
    print("")
    tracks_count += 1
    if tracks_count > MAX_TRACKS:
        break
