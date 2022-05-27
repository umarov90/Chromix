import math

from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention, \
    Add, Embedding, Layer, Reshape, Dropout, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization, LocallyConnected1D
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

dropout_rate = 0.0
leaky_alpha = 0.2
num_patches = 2101
num_filters = 885

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
    triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
    unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
    return tf.gather(unroll_repr, triu_index, axis=1)

  def get_config(self):
    config = super().get_config().copy()
    config['diagonal_offset'] = self.diagonal_offset
    return config


def human_model(input_size, num_features, num_regions, hic_num, hic_size, bin_size, one_d_heads):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    resnet_output = resnet(x, input_size, bin_size)
    our_resnet = Model(inputs, resnet_output, name="our_resnet")

    hic_input = Input(shape=(num_patches, num_filters))
    hi = Conv1D(128, kernel_size=1, strides=1, name="hic_layer_1")(hic_input)
    hx = LeakyReLU(alpha=leaky_alpha, name="hic_act")(hi)
    hxp = tf.keras.layers.AveragePooling1D(pool_size=50)(hi)

    hx = tf.transpose(hx, [0, 2, 1])
    hx = LocallyConnected1D(input_size // 10000, 1, name="hic_layer_2", activation=LeakyReLU(alpha=leaky_alpha))(hx)
    hx = tf.transpose(hx, [0, 2, 1])

    hx = tf.concat([hx, hxp], axis=-1)

    hx = hx[..., 1:-1, :]

    _, seq_len, features = hx.shape

    twod1 = tf.tile(hx, [1, seq_len, 1])
    twod1 = tf.reshape(twod1, [-1, seq_len, seq_len, features])
    twod2 = tf.transpose(twod1, [0, 2, 1, 3])
    twod = tf.concat([twod1, twod2], axis=-1)

    triu = UpperTri()(twod)
    hx = Conv1D(4096, kernel_size=1, strides=1, name="hic_pointwise", activation=LeakyReLU(alpha=leaky_alpha))(triu)
    hx = Conv1D(hic_num, kernel_size=1, strides=1, name="hic_last_conv1d")(hx)
    hx = tf.transpose(hx, [0, 2, 1])
    our_hic = Model(hic_input, hx, name="our_hic")

    all_heads = []
    for key in one_d_heads.keys():
        new_head = make_head(len(one_d_heads[key]), num_regions, "our_" + key)
        all_heads.append(new_head(our_resnet(inputs)))
    all_heads.append(our_hic(our_resnet(inputs)))

    our_model = Model(inputs, all_heads, name="our_model")
    # print("\nModel constructed")
    print(our_model.summary())
    return our_model


def make_head(track_num, num_regions, name):
    head_input = Input(shape=(num_patches, num_filters))
    x = head_input

    # x = tf.transpose(x, [0, 2, 1])
    # # x = Conv1D(num_regions, kernel_size=1, strides=1, use_bias=False, name="regions_projection")(x)
    # x = Dense(input_size // 100, activation=tf.nn.gelu, name="regions_projection")(x)
    # x = tf.transpose(x, [0, 2, 1])

    trim = (x.shape[-2] - num_regions) // 2
    x = x[..., trim:-trim, :]

    x = Dropout(dropout_rate, input_shape=(num_regions, num_filters))(x)
    filter_num = 2048
    # if track_num > 5000:
    #     filter_num *= 2
    x = Conv1D(filter_num, kernel_size=1, strides=1, name=name + "_pointwise", activation=LeakyReLU(alpha=leaky_alpha))(x)
    outputs = Conv1D(track_num, kernel_size=1, strides=1, name=name+"_last_conv1d")(x)
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


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=7,
                 strides=1,
                 activation=True,
                 name="rl_"):
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding="same",
                  name=name + "conv1d")

    x = inputs
    if activation:
        x = LeakyReLU(alpha=leaky_alpha, name=name + "act")(x)
    x = conv(x)
    return x


def resnet(input_x, input_size, bin_size):
    # Initial number of filters
    num_filters = 384

    # First convolutional layer. Since it is first, it is not preceded by activation and batch normalization
    patchify_val = 3
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters,
                     activation=False,
                     strides=patchify_val,
                     kernel_size=15,
                     name="rl_1_")
    current_len = input_size // patchify_val
    # Instantiate the stack of residual units
    num_blocks = 7
    for block in range(num_blocks):
        cname = "rl_" + str(block) + "_"
        strides = 1
        y = x
        # at block num_blocks - 1 final resolution is reached
        if block != num_blocks - 1:
            num_filters = int(num_filters * 1.15)
        activation = True
        if block != 0:
            # Downsample
            strides = 2
            current_len = math.ceil(current_len / 2)
            if block == num_blocks - 1:
                current_len = input_size // bin_size
            if block > 4:
                y = LeakyReLU(alpha=leaky_alpha, name="dwn_" + str(block))(y)
                y = tf.transpose(y, [0, 2, 1])
                # Replace by conv maybe
                y = Dense(current_len, activation=LeakyReLU(alpha=leaky_alpha),
                          name="regions_projection_" + str(block))(y)
                y = tf.transpose(y, [0, 2, 1])
                activation = False
            else:
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 strides=strides,
                                 name="regions_projection_" + str(block))

        # Wide basic block with two CNN layers.
        y = resnet_layer(inputs=y,
                         num_filters=num_filters,
                         activation=activation,
                         name=cname + "1_")
        y = Dropout(dropout_rate)(y)
        y = resnet_layer(inputs=y,
                         num_filters=num_filters,
                         name=cname + "2_")
        if block != num_blocks - 1:
            # linear projection residual shortcut connection to match changed dims
            x = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             activation=False,
                             name=cname + "3_")
        else:
            x = tf.transpose(x, [0, 2, 1])
            x = Conv1D(input_size // bin_size, kernel_size=1, name="linear_regions_projection_" + str(block))(x)
            x = tf.transpose(x, [0, 2, 1])
        x = Add()([x, y])

    # final activation
    x = LeakyReLU(alpha=leaky_alpha, name="res_act_final")(x)
    return x


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
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        #sum with weight
        total_loss = 0.0001 * mse_loss(y_true, y_pred) + 0.001 * pr_loss(y_true, y_pred)
        return total_loss

class mse(tf.keras.losses.Loss):
    def __init__(self, name="mse", **kwargs):
        super().__init__(name=name)
        self.reduction = tf.keras.losses.Reduction.SUM

    def call(self, y_true, y_pred):
        return tf.keras.losses.MSE(y_true,y_pred)

class pearsonr_poisson(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_poisson", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        self.reduction = tf.keras.losses.Reduction.SUM
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.1 
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        pr = pr_loss(y_true, y_pred)
        #poisson part
        poiss_loss = poisson()
        poiss = poiss_loss(y_true, y_pred)
        #sum with weight
        total_loss = (2*pr*poiss)/(pr+poiss)
        return total_loss


class poisson(tf.keras.losses.Loss):
    def __init__(self, name="poisson", **kwargs):
        super().__init__(name=name)
        self.reduction = tf.keras.losses.Reduction.SUM

    def call(self, y_true, y_pred):
        return tf.keras.losses.poisson(y_true, y_pred)


class basenjipearsonr (tf.keras.losses.Loss):
    def __init__(self, name="basenjipearsonr", **kwargs):
        super().__init__(name=name)
        self.reduction = tf.keras.losses.Reduction.SUM
        self.epsilon = 0.000001

    def call(self, y_true, y_pred):
        # y_true = tf.transpose(y_true, [0, 2, 1])
        # y_pred = tf.transpose(y_pred, [0, 2, 1])
        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
        true_sum = tf.reduce_sum(y_true, axis=[0,1])
        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
        pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=[0,1])
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
        m1 = tf.math.sqrt(tf.clip_by_value(true_var, self.epsilon, tf.math.reduce_max(true_var)))
        m2 = tf.math.sqrt(tf.clip_by_value(pred_var, self.epsilon, tf.math.reduce_max(pred_var)))
        tp_var = tf.multiply(m1, m2)
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
