import math

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention, \
    Concatenate, GaussianDropout, GaussianNoise, Add, Embedding, Layer, Dropout, Reshape, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization, LocallyConnected1D, DepthwiseConv1D, DepthwiseConv2D
from tensorflow.keras.models import Model
from keras import backend as K
import numpy as np
import scipy
from numba import jit


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
        hx = Dense(hic_num, name="hic_mlp_2")(hx)
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

    trim = (x.shape[-2] - num_regions) // 2 # 4689 - 1563
    x = x[..., trim:-trim, :]

    outputs = Conv1D(track_num, kernel_size=1, strides=1, name=name + "_last_conv1d")(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    head_output = Activation("linear", dtype='float32')(outputs)
    return Model(head_input, head_output, name=name)


def resnet(input_x, input_size):
    print("Version 1.40")
    # Initial number of filters
    num_filters = 1280
    mlp_start_block = 6
    num_blocks = 12
    patchify_val = 4
    # replace with hardcoded values
    filter_nums = exponential_linspace_int(num_filters, 2 * num_filters, mlp_start_block, divisible_by=128)
    # Patchify layer
    x = Conv1D(num_filters,
               strides=patchify_val,
               kernel_size=patchify_val,
               name="patchify")(input_x)
    x = LayerNormalization(epsilon=1e-6)(x)
    current_len = input_size // patchify_val
    for block in range(num_blocks):
        cname = "body_block_" + str(block) + "_"
        if block != 0 and block < mlp_start_block:
            # Downsample
            num_filters = filter_nums[block]
            strides = 2
            current_len = current_len // strides
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Conv1D(num_filters, kernel_size=strides, strides=strides, padding="same", name=cname + "downsample")(x)
        y = x
        y = GaussianDropout(0.01)(y)
        y1 = DepthwiseConv1D(kernel_size=7, name=cname + "depthwise", padding="same")(y)
        # Spatial MLP for long range interactions
        if block >= mlp_start_block:
            y2 = tf.transpose(y, [0, 2, 1])
            y2 = Dense(current_len // 2, name=cname + "mlp_1")(y2)
            y2 = Activation(tf.nn.gelu)(y2)
            # y2 = MonteCarloDropout(0.1)(y2)
            y2 = Dense(current_len, name=cname + "mlp_2")(y2)
            y2 = tf.transpose(y2, [0, 2, 1])
            y = Concatenate(axis=-1)([y1, y2]) # y1 + y2
        else:
            y = y1
        y = LayerNormalization(epsilon=1e-6)(y)
        # Pointwise to mix the channels
        y = Conv1D(4 * num_filters, kernel_size=1, padding="same", name=cname + "pointwise_1")(y)
        y = Activation(tf.nn.gelu)(y)
        # y = MonteCarloDropout(0.1)(y)
        y = Conv1D(num_filters, kernel_size=1, padding="same", name=cname + "pointwise_2")(y)
        x = x + y

    x = LayerNormalization(epsilon=1e-6)(x)
    x = Conv1D(2 * num_filters, kernel_size=1, name="body_output", activation=tf.nn.gelu, dtype='float32')(x)
    return x


@tf.function
def fast_mse01(y_true, y_pred):
    normal_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true > 0))
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true > 0))
    non_zero_mse = tf.reduce_mean(tf.square(y_true_positive - y_pred_positive))
    non_zero_mse = tf.where(tf.math.is_nan(non_zero_mse), tf.zeros_like(non_zero_mse), non_zero_mse)
    total_loss = normal_mse + 0.1 * non_zero_mse
    return total_loss


@tf.function
def fast_mse1(y_true, y_pred):
    normal_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true > 0))
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true > 0))
    non_zero_mse = tf.reduce_mean(tf.square(y_true_positive - y_pred_positive))
    non_zero_mse = tf.where(tf.math.is_nan(non_zero_mse), tf.zeros_like(non_zero_mse), non_zero_mse)
    total_loss = normal_mse + 1 * non_zero_mse
    return total_loss

@tf.function
def fast_mse5(y_true, y_pred):
    normal_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true > 0))
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true > 0))
    non_zero_mse = tf.reduce_mean(tf.square(y_true_positive - y_pred_positive))
    non_zero_mse = tf.where(tf.math.is_nan(non_zero_mse), tf.zeros_like(non_zero_mse), non_zero_mse)
    total_loss = normal_mse + 2 * non_zero_mse
    return total_loss

@tf.function
def mse_plus_cor(y_true, y_pred):
    normal_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    y_pred_positive = tf.gather_nd(y_pred, tf.where(y_true > 0))
    y_true_positive = tf.gather_nd(y_true, tf.where(y_true > 0))

    non_zero_mse = tf.reduce_mean(tf.square(y_true_positive - y_pred_positive))
    non_zero_mse = tf.where(tf.math.is_nan(non_zero_mse), tf.zeros_like(non_zero_mse), non_zero_mse)

    cor = cor_loss(y_pred_positive, y_true_positive, -1)
    total_loss = normal_mse + 0.2 * non_zero_mse + 0.001 * tf.math.reduce_mean(cor)

    return total_loss


@tf.function
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
            predictions = np.concatenate((predictions, p1))
    return predictions


def batch_predict_effect(p, model, seqs1, seqs2):
    body = model.get_layer("our_resnet")
    print(f"Info {p.predict_batch_size} {p.w_step}")
    n_times = 1
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        p1s = []
        for i in range(n_times):
            p1s.append(body.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p1 = np.mean(p1s, axis=0)
        # pr = model.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size))
        # p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        p2s = []
        for i in range(n_times):
            p2s.append(body.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p2 = np.mean(p2s, axis=0)
        expression = p1[:, :, p.mid_bin] 
        # expression = np.squeeze(expression)
        expression = np.max(expression, axis=1, keepdims=True)

        fold_change = p2[:, :, p.mid_bin] / p1[:, :, p.mid_bin] 
        # fold_change = np.squeeze(fold_change)
        fold_change = np.max(fold_change, axis=1, keepdims=True)
        # pr = model.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size))
        # p2 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        effect = np.max(np.abs(p1 - p2), axis=-1)
        # effect = np.mean(p1, axis=-1) - np.mean(p2, axis=-1)
        # effect1 = np.mean(p1 - p2, axis=-1)
        # effect2 = np.mean(p1 - p2, axis=-2)
        # effect3 = np.max(np.abs(p1 - p2), axis=-1)
        # effect4 = np.max(np.abs(p1 - p2), axis=-2)
        # effect = np.concatenate((effect3, effect4), axis=-1)
        if w == 0:
            predictions = effect
            fold_changes = fold_change
            expressions = expression
        else:
            predictions = np.concatenate((predictions, effect))
            fold_changes = np.concatenate((fold_changes, fold_change))
            expressions = np.concatenate((expressions, expression))
    fold_changes = np.clip(fold_changes, 0, 100)
    fold_changes = np.log(fold_changes + 1)
    fold_changes[np.isnan(fold_changes)] = -1
    return predictions, fold_changes


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def cross_entropy(p, q):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    q = np.where(q>1.0e-10,q,1.0e-10) #fill the zeros with 10**-10
    return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])


# def JS_divergence(p,q):
#     M=(p+q)/2
#     return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)


# def KL_divergence(p,q):
#     return scipy.stats.entropy(p,q)


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def normalization(data):
    _range=np.max(data)-np.min(data)
    return (data-np.min(data))/_range


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def fast_ce(p1, p2):
    tmp1=[]
    for i in range(p1.shape[0]):
        tmp2=[]
        for j in range(p1.shape[1]):
            #tmp2.append(JS_divergence(normalization(p1[i][j]),normalization(p2[i][j])))
            #tmp2.append(scipy.stats.entropy(p1[i][j],p2[i][j],base=2))
            tmp2.append(cross_entropy(normalization(p1[i][j]),normalization(p2[i][j])))
        tmp1.append(tmp2)
    return np.array(tmp1)

def batch_predict_effect_x(p, model, seqs1, seqs2):
    body = model.get_layer("our_resnet")
    n_times = 1
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        p1s = []
        for i in range(n_times):
            p1s.append(body.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p1 = np.mean(p1s, axis=0)
        # pr = model.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size))
        # p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        p2s = []
        for i in range(n_times):
            p2s.append(body.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p2 = np.mean(p2s, axis=0)
        expression = p1[:, :, p.mid_bin] 
        # expression = np.squeeze(expression)
        expression = np.max(expression, axis=1, keepdims=True)
        fold_change = p2[:, :, p.mid_bin] / p1[:, :, p.mid_bin] 
        # fold_change = np.squeeze(fold_change)
        fold_change = np.max(fold_change, axis=1, keepdims=True)
        # pr = model.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size))
        # p2 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        #effect = np.max(np.abs(p1 - p2), axis=-1)
        effect = fast_ce(p1, p2)
        if w == 0:
            predictions = effect
            fold_changes = fold_change
            expressions = expression
        else:
            predictions = np.concatenate((predictions, effect))
            fold_changes = np.concatenate((fold_changes, fold_change))
            expressions = np.concatenate((expressions, expression))
    fold_changes = np.clip(fold_changes, 0, 100)
    fold_changes = np.log(fold_changes + 1)
    fold_changes[np.isnan(fold_changes)] = -1
    return predictions, fold_changes


def batch_predict_effect_long(p, model, seqs1, seqs2):
    body = model.get_layer("our_resnet")
    n_times = 2
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        p1s = []
        for i in range(n_times):
            p1s.append(body.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p1 = np.mean(p1s, axis=0)
        p2s = []
        for i in range(n_times):
            p2s.append(body.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p2 = np.mean(p2s, axis=0)
        long_range = np.abs(p1 - p2)
        mid = long_range.shape[-2] // 2
        long_range = np.delete(long_range, np.s_[mid - 24:mid + 24], -2)
        effect = np.max(long_range, axis=-1)
        if w == 0:
            predictions = effect
            # print(f"mid {mid} shape {long_range.shape}")
        else:
            predictions = np.concatenate((predictions, effect))
    return predictions


def batch_predict_effect2(p, model, seqs1, seqs2): # , inds
    n_times = 1
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        # print(f"seqs shape {np.asarray(seqs1[w:w + p.w_step]).shape}")
        p1s = []
        for i in range(n_times):
            p1s.append(model.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        # print(f"p1s {np.asarray(p1s).shape}")
        p1 = np.mean(p1s, axis=0)
        # p1 = p1[:, inds, :]
        # print(f"p1 {p1.shape}")
        p2s = []
        for i in range(n_times):
            p2s.append(model.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p2 = np.mean(p2s, axis=0)
        # p2 = p2[:, inds, :]

        effect = np.max(np.abs(p1 - p2), axis=-2)
        # print(f"effect {effect.shape}")
        if w == 0:
            predictions = effect
        else:
            predictions = np.concatenate((predictions, effect), dtype=np.float16)
    return predictions

def batch_predict_effect_linking(p, model, seqs1, seqs2, tss_positions):
    body = model.get_layer("our_resnet")
    n_times = 3
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        p1s = []
        for i in range(n_times):
            p1s.append(body.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p1 = np.mean(p1s, axis=0)
        p_cutout = []
        for j in range(len(p1)):
            p_cutout.append(np.concatenate((p1[j][len(p1[j]) // 2 - 10: len(p1[j]) // 2 + 10], p1[j][tss_positions[w + j] - 10: tss_positions[w + j] + 10])))
        p1 = np.asarray(p_cutout)
        p2s = []
        for i in range(n_times):
            p2s.append(body.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose = 0))
        p2 = np.mean(p2s, axis=0)
        p_cutout = []
        for j in range(len(p2)):
            p_cutout.append(np.concatenate((p2[j][len(p2[j]) // 2 - 10: len(p2[j]) // 2 + 10], p2[j][tss_positions[w + j] - 10: tss_positions[w + j] + 10])))
        p2 = np.asarray(p_cutout)
        effect = np.max(np.abs(p1 - p2), axis=-1)
        if w == 0:
            predictions = effect
        else:
            predictions = np.concatenate((predictions, effect))
    return predictions
    

# from https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py
def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base ** i) for i in range(num)]

class MonteCarloDropout(Dropout):
    def call(self, inputs):
      return super().call(inputs, training=True)
