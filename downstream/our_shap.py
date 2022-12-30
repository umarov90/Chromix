import shap
import numpy as np
import os
import pathlib
import model as mo
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main_params import MainParams
import parse_data as parser


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


p = MainParams()
one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")
def get_seq(info, input_size, sub_half=False):
    start = int(info[1] - (info[1] % p.bin_size) - input_size // 2)
    if sub_half:
        start = start - p.bin_size // 2
    extra = start + input_size - len(one_hot[info[0]])
    if start < 0:
        ns = one_hot[info[0]][0:start + input_size]
        ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
    elif extra > 0:
        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
        ns = np.concatenate((ns, np.zeros((extra, 5))))
    else:
        ns = one_hot[info[0]][start:start + input_size]
    return ns[:, :-1]

our_model = tf.keras.models.load_model('test.h5')
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
# heads = joblib.load(f"{p.pickle_folder}heads.gz")
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads["expression"])
#     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
# our_model.save('test.h5', include_optimizer=False, save_format="h5")
# exit()
test_seq = []
_, _, test_info, _ = parser.parse_sequences(p)
for index, info in enumerate(test_info):
    seq = get_seq([info[0], info[1]], p.input_size)
    test_seq.append(seq)
test_seq = np.asarray(test_seq, dtype=bool)
# select a set of background examples to take an expectation over
background = test_seq[np.random.choice(test_seq.shape[0], 2, replace=False)]

e = shap.DeepExplainer((our_model.input, our_model.output[:, :, p.mid_bin]), background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
seqs_to_explain = test_seq[:2]
shap_values = e.shap_values(seqs_to_explain)
np.savez_compressed("ohe.npz", seqs_to_explain)
np.savez_compressed("shap.npz", shap_values)

