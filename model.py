import math

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention, \
    Add, Embedding, Layer, Reshape, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization, LocallyConnected1D, DepthwiseConv1D, DepthwiseConv2D
from tensorflow.keras.models import Model
from keras import backend as K
import numpy as np


def make_model(input_size, num_features, num_regions, hic_num, hic_size, one_d_heads):
    inputs = Input(shape=(input_size, num_features))
    x = inputs
    output1d = resnet(x, input_size)
    our_resnet = Model(inputs, output1d, name="our_resnet")
    outs = our_resnet(inputs)

    if hic_num > 0:
        # Make hic head
        seq_len = output1d.shape[-2]
        features = output1d.shape[-1]
        hic_input = Input(shape=(seq_len, features))

        hx = hic_input
        hx = tf.transpose(hx, [0, 2, 1])
        hx = Dense(hic_size, activation=tf.nn.gelu, name="hic_mlp_1")(hx)
        hx = tf.transpose(hx, [0, 2, 1])
        hx = Dense(4096, activation=tf.nn.gelu, name="hic_mlp_2")(hx)
        hx = Dense(hic_num, name="hic_mlp_3")(hx)
        hx = tf.transpose(hx, [0, 2, 1])

        hx = Activation('linear', dtype='float32')(hx)
        our_hic = Model(hic_input, hx, name="our_hic")

    all_heads = []
    if isinstance(one_d_heads, dict):
        for key in one_d_heads.keys():
            new_head = make_head(len(one_d_heads[key]), num_regions, output1d, "our_" + key)
            all_heads.append(new_head(outs))
    else:
        new_head = make_head(len(one_d_heads), num_regions, output1d, "our_expression")
        all_heads.append(new_head(outs))

    if hic_num > 0:
        all_heads.append(our_hic(outs))

    our_model = Model(inputs, all_heads, name="our_model")
    # print("\nModel constructed")
    print(our_model.summary())
    return our_model


def make_head(track_num, num_regions, output1d, name):
    seq_len = output1d.shape[-2]
    features = output1d.shape[-1]
    head_input = Input(shape=(seq_len, features))
    x = head_input

    trim = (x.shape[-2] - num_regions) // 2
    x = x[..., trim:-trim, :]

    filter_num = 2048
    activation = "linear"
    if name == "our_expression":
        filter_num *= 2
    x = Conv1D(filter_num, kernel_size=1, strides=1, name=name + "_pointwise", activation=tf.nn.gelu)(x)
    outputs = Conv1D(track_num, kernel_size=1, strides=1, name=name + "_last_conv1d")(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    head_output = Activation(activation, dtype='float32')(outputs)
    return Model(head_input, head_output, name=name)


def resnet(input_x, input_size):
    # Initial number of filters
    num_filters = 512
    mlp_start_block = 2
    mlp_hidden_dim_reduction = 32
    num_blocks = 6
    patchify_val = 4
    filter_nums = exponential_linspace_int(num_filters, 1024, num_blocks, divisible_by=64)
    # Patchify layer
    x = Conv1D(num_filters,
               strides=patchify_val,
               kernel_size=patchify_val,
               name="patchify")(input_x)
    x = LayerNormalization(dtype=tf.float32)(x)
    current_len = input_size // patchify_val
    for block in range(num_blocks):
        cname = "body_block_" + str(block) + "_"
        strides = 1
        y = x
        if block != 0:
            # Downsample
            strides = 2
            current_len = math.ceil(current_len / strides)
            y = LayerNormalization(dtype=tf.float32)(y)
            y = DepthwiseConv1D(kernel_size=strides,
                                strides=strides,
                                padding="same",
                                name=cname + "downsample")(y)
            if block > mlp_start_block and mlp_hidden_dim_reduction > 4:
                mlp_hidden_dim_reduction = mlp_hidden_dim_reduction // 2

        num_filters = filter_nums[block]

        y1 = y
        y1 = DepthwiseConv1D(kernel_size=7, name=cname + "depthwise", padding="same")(y1)
        # Spatial MLP for long range interactions
        if block >= mlp_start_block:
            y2 = y
            y2 = tf.transpose(y2, [0, 2, 1])
            hd = current_len // mlp_hidden_dim_reduction
            y2 = Dense(hd, activation=tf.nn.gelu, name=cname + "mlp_1")(y2)
            y2 = Dense(current_len, name=cname + "mlp_2")(y2)
            y2 = tf.transpose(y2, [0, 2, 1])
            y = y1 + y2
        else:
            y = y1
        y = LayerNormalization(dtype=tf.float32)(y)
        # Pointwise to mix the channels
        y = Conv1D(4 * num_filters,
                   kernel_size=1, padding="same",
                   name=cname + "pointwise_1")(y)
        y = tf.keras.layers.Activation(tf.nn.gelu)(y)
        y = Conv1D(num_filters, kernel_size=1, padding="same",
                   name=cname + "pointwise_2")(y)

        # linear projection residual shortcut connection
        x = Conv1D(num_filters,
                   kernel_size=7,
                   strides=strides,
                   padding="same",
                   name=cname + "linear_projection")(x)

        x = x + y
    return LayerNormalization(dtype=tf.float32)(x)


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


def batch_predict(p, model, seqs):
    for w in range(0, len(seqs), p.w_step):
        print(w, end=" ")
        pr = model.predict(wrap2(seqs[w:w + p.w_step], p.predict_batch_size))
        p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
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
            predictions = np.concatenate((predictions, p1), dtype=np.float16)
    return predictions


def batch_predict_effect(p, model, seqs1, seqs2):
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        pr = model.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size))
        p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        pr = model.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size))
        p2 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        # if len(hic_keys) > 0:
        #     p2 = p1[0][:, :, p.mid_bin - 1] + p1[0][:, :, p.mid_bin] + p1[0][:, :, p.mid_bin + 1]
        #     if w == 0:
        #         predictions = p2
        #     else:
        #         predictions = np.concatenate((predictions, p2), dtype=np.float32)
        # else:
        effect = np.max(np.abs(p1 - p2), axis=-1)
        if w == 0:
            predictions = effect
        else:
            predictions = np.concatenate((predictions, effect), dtype=np.float16)
    return predictions


def mse_plus_cor(y_true, y_pred):
    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true > 0))
    print(f"Shape is {y_pred_positive.shape}")
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true > 0))

    cor = cor_loss(y_pred_positive, y_true_positive, -1)

    total_loss = fast_mse(y_true, y_pred) + 0.0001 * tf.math.reduce_mean(cor)

    return total_loss


def cor_loss(x, y, cor_axis):
    mx = tf.math.reduce_mean(x, axis=cor_axis)
    my = tf.math.reduce_mean(y, axis=cor_axis)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(xm * ym, axis=cor_axis)
    r_den = tf.math.reduce_std(xm, axis=cor_axis) * tf.math.reduce_std(ym, axis=cor_axis)
    r_den = r_den + 0.0000001
    r = r_num / r_den
    r = tf.math.maximum(tf.math.minimum(r, 1.0), -1.0)
    cor = 1 - r
    cor = tf.where(tf.math.is_nan(cor), tf.zeros_like(cor), cor)
    return cor


@tf.function
def fast_mse(y_true, y_pred):
    y_pred_negative = tf.gather_nd(y_pred, tf.where(y_true==0))
    y_true_negative = tf.gather_nd(y_true, tf.where(y_true==0))
    zero_mse = tf.reduce_mean(tf.square(y_pred_negative - y_true_negative))
    zero_mse = tf.where(tf.math.is_nan(zero_mse), tf.zeros_like(zero_mse), zero_mse)

    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true>0))
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true>0))
    non_zero_mse = tf.reduce_mean(tf.square(y_true_positive - y_pred_positive))
    non_zero_mse = tf.where(tf.math.is_nan(non_zero_mse), tf.zeros_like(non_zero_mse), non_zero_mse)

    total_loss = 0.9 * zero_mse + 0.1 * non_zero_mse

    return total_loss


@tf.function
def fast_mse_hic(y_true, y_pred):
    normal_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true>0))
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true>0))
    non_zero_mse = tf.reduce_mean(tf.square(y_true_positive - y_pred_positive))
    non_zero_mse = tf.where(tf.math.is_nan(non_zero_mse), tf.zeros_like(non_zero_mse), non_zero_mse)
    total_loss = normal_mse + 0.2 * non_zero_mse
    return total_loss

# from https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py
def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]


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

# taken from https://stackoverflow.com/questions/46881006/slicing-tensor-with-list-tensorflow
def list_slice(tensor, indices, axis):
    """
    Args
    ----
    tensor (Tensor) : input tensor to slice
    indices ( [int] ) : list of indices of where to perform slices
    axis (int) : the axis to perform the slice on
    """

    slices = []   

    ## Set the shape of the output tensor. 
    # Set any unknown dimensions to -1, so that reshape can infer it correctly. 
    # Set the dimension in the slice direction to be 1, so that overall dimensions are preserved during the operation
    shape = tensor.get_shape().as_list()
    shape[shape==None] = -1
    shape[axis] = 1

    nd = len(shape)

    for i in indices:   
        _slice = [slice(None)]*nd
        _slice[axis] = slice(i,i+1)
        slices.append(tf.reshape(tensor[_slice], shape))

    return tf.concat(slices, axis=axis)
