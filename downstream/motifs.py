import os
import gc
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
from pathlib import Path
import random
import common as cm

MAX_PROMOTERS = 3000
p = MainParams()
head = joblib.load(f"{p.pickle_folder}heads.gz")
head_tracks = head["hg38"]["expression"]
one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head_tracks)
#     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))

train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
eval_infos = valid_info + test_info
eval_infos = random.sample(eval_infos, len(eval_infos))

test_seq = []
for info in eval_infos:
    start = info[1] - p.half_size
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

attributions = []
att_len = 20
seqs_to_explain = []
fasta = []
for si, seq in enumerate(test_seq):
    if eval_infos[si][4] != "+":
        continue
    if si % 1 == 0:
        print(f"Promoter {si}")
    if si > MAX_PROMOTERS:
        break
    fasta.append(f"> Promoter {si}")
    seqs_to_explain.append(seq[p.half_size - att_len // 2: p.half_size + att_len // 2, :])
    fasta.append(cm.to_dna(seq[p.half_size - att_len // 2: p.half_size + att_len // 2, :]))
    continue
    # attribution
    batch = []
    # for i in range(p.half_size - att_len // 2, p.half_size + att_len // 2, 1):
    #     for b in range(4):
    #         seq2 = seq.copy()
    #         seq2[i] = np.asarray([0, 0, 0, 0])
    #         seq2[i][b] = 1
    #         batch.append(seq2)
    # p.w_step = 800
    # for w in range(0, len(batch), p.w_step):
    #     print(w, end=" ")
    #     pr = our_model.predict(mo.wrap2(batch[w:w + p.w_step], p.predict_batch_size), verbose=0)
    #     pr = pr[:, :, p.mid_bin - 1] + pr[:, :, p.mid_bin] + pr[:, :, p.mid_bin + 1]
    #     if w == 0:
    #         predictions = pr
    #     else:
    #         predictions = np.concatenate((predictions, pr))
    # gc.collect()
    # attribution = np.zeros((att_len, 4, len(head_tracks)))
    # for i in range(0, len(predictions), 4):
    #     mean = np.mean(predictions[i: i + 4], axis=0)
    #     for b in range(4):
    #         attribution[i // 4, b] = predictions[i + b] - mean
    # x = np.mean(np.sum(np.abs(attribution), axis=1), axis=1)
    # print(np.argmax(x))
    # attributions.append(attribution)

with open("attributions/seqs.fa", 'w+') as f:
    f.write('\n'.join(fasta))
exit()
    
attributions = np.asarray(attributions)      
Path("attributions").mkdir(parents=True, exist_ok=True)
seqs_to_explain = np.asarray(seqs_to_explain)
np.savez_compressed("attributions/ohe.npz", np.swapaxes(seqs_to_explain,1,2))
for track_to_use, track in enumerate(head_tracks):
    if "FANTOM5" not in track or "K562" not in track or "response" in track:
        continue
    np.savez_compressed(f"attributions/attribution_{track}.npz",
                        np.swapaxes(np.asarray(attributions[:, :, :, track_to_use]),1,2))