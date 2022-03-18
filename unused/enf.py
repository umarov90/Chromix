import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
import tensorflow_hub as hub
import pathlib


script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/data_dirs").read().strip().split("\n")
os.chdir(folders[0])

enformer = tf.saved_model.load("enformer_1")

SEQ_LENGTH = 393_216

# Numpy array [batch_size, SEQ_LENGTH, 4] one hot encoded in order 'ACGT'. The
# `one_hot_encode` function is available in `enformer_usage.py` and outputs can be
# stacked to form a batch.
inputs = tf.zeros((1, SEQ_LENGTH, 4), dtype=tf.float32)
predictions = enformer.call(inputs)
predictions['human'].shape  # [batch_size, 896, 5313]
predictions['mouse'].shape  # [batch_size, 896, 1643]