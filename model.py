import math

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention, \
    Add, Embedding, Layer, Reshape, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization, LocallyConnected1D, DepthwiseConv1D, DepthwiseConv2D
from tensorflow.keras.models import Model

import numpy as np


# Taken from https://github.com/calico/basenji/blob/master/basenji/layers.py
class UpperTri(tf.keras.layers.Layer):
    ''' Unroll matrix to its upper triangular portion.'''

    def __init__(self, diagonal_offset=1):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def call(self, inputs):
        seq_len = inputs.shape[1]
        output_dim = inputs.shape[-1]

        if type(seq_len) == tf.compat.v1.Dimension:
            seq_len = seq_len.value
            output_dim = output_dim.value

        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = list(triu_tup[0] + seq_len * triu_tup[1])
        unroll_repr = tf.reshape(inputs, [-1, seq_len ** 2, output_dim])
        return tf.gather(unroll_repr, triu_index, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config['diagonal_offset'] = self.diagonal_offset
        return config


def human_model(input_size, num_features, num_regions, hic_num, bin_size, hic_bin_size, one_d_heads):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    output1d, output2d = resnet(x, input_size, bin_size, hic_bin_size)
    our_resnet = Model(inputs, [output1d, output2d], name="our_resnet")

    seq_len = output2d.shape[-2]
    features = output2d.shape[-1]
    hic_input = Input(shape=(seq_len, features))

    hx = hic_input

    twod1 = tf.tile(hx, [1, seq_len, 1])
    twod1 = tf.reshape(twod1, [-1, seq_len, seq_len, features])
    twod2 = tf.transpose(twod1, [0, 2, 1, 3])
    twod = tf.concat([twod1, twod2], axis=-1)

    twod = DepthwiseConv2D(kernel_size=9, name="hic_dw", padding="same")(twod)

    twod = twod[..., 5:-5, 5:-5, :]

    triu = UpperTri()(twod)
    hx = Conv1D(4096, kernel_size=1, strides=1, name="hic_pointwise", activation=tf.nn.gelu)(triu)
    hx = Conv1D(hic_num, kernel_size=1, strides=1, name="hic_last_conv1d")(hx)
    hx = tf.transpose(hx, [0, 2, 1])
    our_hic = Model(hic_input, hx, name="our_hic")

    all_heads = []
    outs = our_resnet(inputs)
    for key in one_d_heads.keys():
        new_head = make_head(len(one_d_heads[key]), num_regions, output1d, "our_" + key)
        all_heads.append(new_head(outs[0]))
    all_heads.append(our_hic(outs[1]))

    our_model = Model(inputs, all_heads, name="our_model")
    # print("\nModel constructed")
    print(our_model.summary())
    return our_model


def make_head(track_num, num_regions, output1d, name):
    seq_len = output1d.shape[-2]
    features = output1d.shape[-1]
    head_input = Input(shape=(seq_len, features))
    x = head_input

    x = DepthwiseConv1D(kernel_size=9, strides=1, name=name + "_dw", activation=tf.nn.gelu, padding="same")(x)

    trim = (x.shape[-2] - num_regions) // 2
    x = x[..., trim:-trim, :]

    filter_num = 2048
    if name == "our_expression":
        filter_num *= 2
    x = Conv1D(filter_num, kernel_size=1, strides=1, name=name + "_pointwise", activation=tf.nn.gelu)(x)
    outputs = Conv1D(track_num, kernel_size=1, strides=1, name=name + "_last_conv1d")(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    # print(outputs)
    head_output = outputs
    return Model(head_input, head_output, name=name)


def small_model(input_size, num_features, num_regions, cell_num, bin_size):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    resnet_output = resnet(x, input_size, bin_size)
    our_resnet = Model(inputs, resnet_output, name="our_resnet")

    new_head = make_head(cell_num, num_regions, "our_expression")

    our_model = Model(inputs, new_head(our_resnet(inputs)), name="our_model")
    # print("\nModel constructed")
    # print(our_model.summary())
    return our_model

# from https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py
def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]

def resnet(input_x, input_size, bin_size, hic_bin_size):
    # Initial number of filters
    num_filters = 128
    mlp_start_block = 6
    mlp_hidden_dim_reduction = 8
    mlp_size = 256

    # First convolutional layer. Since it is first, it is not preceded by activation and batch normalization
    patchify_val = 4
    x = Conv1D(num_filters,
               strides=patchify_val,
               kernel_size=patchify_val,
               name="rl_1_")(input_x)
    x = LayerNormalization()(x)
    current_len = input_size // patchify_val
    # Instantiate the stack of residual units
    num_blocks = 14
    filter_nums = exponential_linspace_int(num_filters, 1024, 6, divisible_by=64)
    for block in range(num_blocks):
        cname = "rl_" + str(block) + "_"
        strides = 1
        y = x
        if block != 0:
            # Downsample
            if block < mlp_start_block:
                strides = 2
            elif block < mlp_start_block + 4:
                strides = 1
            else:
                strides = 3
                if mlp_hidden_dim_reduction > 1:
                    mlp_hidden_dim_reduction = mlp_hidden_dim_reduction / 2
            if strides > 1:
                current_len = math.ceil(current_len / strides)
                y = LayerNormalization()(y)
                y = DepthwiseConv1D(kernel_size=strides,
                                    strides=strides,
                                    padding="same",
                                    name="downsample_" + str(block))(y)

        if block < mlp_start_block:
            num_filters = filter_nums[block]

        if block >= mlp_start_block:
            y1 = y[..., :-mlp_size]
        else:
            y1 = y
        y1 = DepthwiseConv1D(kernel_size=9, name="depthwise_" + str(block), padding="same")(y1)
        if block >= mlp_start_block:
            y2 = y[..., -mlp_size:]
            y2 = tf.transpose(y2, [0, 2, 1])
            hd = current_len // mlp_hidden_dim_reduction
            y2 = Dense(hd, activation=tf.nn.gelu, name="mlp_1_" + str(block))(y2)
            y2 = Dense(current_len, name="mlp_2_" + str(block))(y2)
            y2 = tf.transpose(y2, [0, 2, 1])
            y = tf.concat([y1, y2], axis=-1)
        else:
            y = y1
        y = LayerNormalization()(y)
        # Wide basic block with two CNN layers.
        y = Conv1D(4 * num_filters,
                   kernel_size=1, padding="same",
                   name=cname + "1_")(y)
        y = tf.keras.layers.Activation(tf.nn.gelu)(y)
        y = Conv1D(num_filters, kernel_size=1, padding="same",
                   name=cname + "2_")(y)

        # linear projection residual shortcut connection
        x = Conv1D(num_filters,
                   kernel_size=1,
                   strides=strides,
                   padding="same",
                   name=cname + "3_")(x)

        x = Add()([x, y])
        if block == mlp_start_block + 4 - 1:
            output_1d = LayerNormalization()(x)
    return output_1d, LayerNormalization()(x)


def wrap(input_sequences, output_scores, bs):
    with tf.device('cpu:0'):
        train_data = tf.data.Dataset.from_tensor_slices((input_sequences, output_scores))
        train_data = train_data.batch(bs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        return train_data


def wrap_with_hic(input_sequences, output_scores, bs):
    with tf.device('cpu:0'):
        train_data = tf.data.Dataset.from_tensor_slices(
            (input_sequences, {"our_head": output_scores[0], "our_hic": output_scores[1]}))
        train_data = train_data.batch(bs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        return train_data


def wrap_for_human_training(input_sequences, output_scores, bs):
    with tf.device('cpu:0'):
        train_data = tf.data.Dataset.from_tensor_slices((input_sequences, output_scores))
        train_data = train_data.batch(bs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        return train_data


def wrap2(input_sequences, bs):
    with tf.device('cpu:0'):
        train_data = tf.data.Dataset.from_tensor_slices(input_sequences)
        train_data = train_data.batch(bs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        return train_data


def batch_predict(model, seqs):
    w_step = 500
    predict_batch_size = 8
    for w in range(0, len(seqs), w_step):
        print(w, end=" ")
        p1 = model.predict(wrap2(seqs[w:w + w_step], predict_batch_size))
        # if len(hic_keys) > 0:
        #     p2 = p1[0][:, :, p.mid_bin - 1] + p1[0][:, :, p.mid_bin] + p1[0][:, :, p.mid_bin + 1]
        #     if w == 0:
        #         predictions = p2
        #     else:
        #         predictions = np.concatenate((predictions, p2), dtype=np.float32)
        # else:
        if w == 0:
            predictions = p1
        else:
            predictions = np.concatenate((predictions, p1), dtype=np.float32)
    return predictions


# taken from https://github.com/shtoneyan/gopher/blob/main/gopher/losses.py
class pearsonr_mse(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_mse", **kwargs):
        super().__init__(name=name)
        self.reduction = tf.keras.losses.Reduction.SUM

    def call(self, y_true, y_pred):
        # multinomial part of loss function
        pr_loss = basenjipearsonr()
        mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        # sum with weight
        total_loss = 0.0001 * mse_loss(y_true, y_pred) + 0.001 * pr_loss(y_true, y_pred)
        return total_loss


class basenjipearsonr(tf.keras.losses.Loss):
    def __init__(self, name="basenjipearsonr", **kwargs):
        super().__init__(name=name)
        self.reduction = tf.keras.losses.Reduction.SUM
        self.epsilon = 0.000001

    def call(self, y_true, y_pred):
        # y_true = tf.transpose(y_true, [0, 2, 1])
        # y_pred = tf.transpose(y_pred, [0, 2, 1])
        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1])
        true_sum = tf.reduce_sum(y_true, axis=[0, 1])
        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0, 1])
        pred_sum = tf.reduce_sum(y_pred, axis=[0, 1])
        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0, 1])
        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=[0, 1])
        true_mean = tf.divide(true_sum, count)
        true_mean2 = tf.math.square(true_mean)
        pred_mean = tf.divide(pred_sum, count)
        pred_mean2 = tf.math.square(pred_mean)

        term1 = product
        term2 = -tf.multiply(true_mean, pred_sum)
        term3 = -tf.multiply(pred_mean, true_sum)
        term4 = tf.multiply(count, tf.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = true_sumsq - tf.multiply(count, true_mean2)
        pred_var = pred_sumsq - tf.multiply(count, pred_mean2)

        tp_var = tf.multiply(true_var, pred_var)
        correlation = tf.divide(covariance, tp_var + self.epsilon)
        correlation = tf.clip_by_value(correlation, -1, 1)
        correlation = tf.where(tf.math.is_nan(correlation), tf.zeros_like(correlation), correlation)
        correlation = 1 - correlation
        return tf.reduce_mean(correlation)


def batch_predict_effect(model, seqs1, seqs2):
    w_step = 500
    predict_batch_size = 8
    for w in range(0, len(seqs1), w_step):
        print(w, end=" ")
        p1 = model.predict(wrap2(seqs1[w:w + w_step], predict_batch_size))
        p2 = model.predict(wrap2(seqs2[w:w + w_step], predict_batch_size))
        # if len(hic_keys) > 0:
        #     p2 = p1[0][:, :, p.mid_bin - 1] + p1[0][:, :, p.mid_bin] + p1[0][:, :, p.mid_bin + 1]
        #     if w == 0:
        #         predictions = p2
        #     else:
        #         predictions = np.concatenate((predictions, p2), dtype=np.float32)
        # else:
        effect = np.mean(p1 - p2, axis=-1)
        if w == 0:
            predictions = effect
        else:
            predictions = np.concatenate((predictions, effect), dtype=np.float32)
    return predictions
