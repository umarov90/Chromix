import math

from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention, \
    Add, Embedding, Layer, Reshape, Dropout, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

dropout_rate = 0.3
leaky_alpha = 0.2


def hic_model(input_size, num_features, num_regions, cell_num, hic_num, hic_size):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    resnet_output = resnet(x, input_size, 200)
    our_resnet = Model(inputs, resnet_output, name="our_resnet")
    num_patches = 6563
    num_filters = 904

    hic_input = Input(shape=(num_patches, num_filters))
    hx = Conv1D(128, kernel_size=1, strides=1, name="pointwise_hic_1", activation=tf.nn.gelu)(hic_input)

    hx = tf.transpose(hx, [0, 2, 1])
    hx = Conv1D(input_size // 1000, kernel_size=1, strides=1, name="pointwise_hic_2", activation=tf.nn.gelu)(hx)
    hx = tf.transpose(hx, [0, 2, 1])

    hx = Flatten()(hx)
    h_layers = []
    for h in range(hic_num):
        h_layers.append(Dense(hic_size)(hx))
    hic_output = tf.stack(h_layers, axis=1)
    print(hic_output)
    our_hic = Model(hic_input, hic_output, name="our_hic")

    head_input = Input(shape=(num_patches, num_filters))
    x = head_input

    x = tf.transpose(x, [0, 2, 1])
    # x = Conv1D(num_regions, kernel_size=1, strides=1, use_bias=False, name="regions_projection")(x)
    x = Dense(num_regions, activation=tf.nn.gelu, name="regions_projection")(x)
    x = tf.transpose(x, [0, 2, 1])

    x = Dropout(dropout_rate, input_shape=(num_regions, num_filters))(x)

    x = Conv1D(2048, kernel_size=1, strides=1, name="pointwise", activation=tf.nn.gelu)(x)
    outputs = Conv1D(cell_num, kernel_size=1, strides=1, name="last_conv1d")(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    print(outputs)
    head_output = outputs
    our_head = Model(head_input, head_output, name="our_head")
    print(our_head)

    our_model = Model(inputs, [our_head(our_resnet(inputs)),
                               our_hic(our_resnet(inputs))], name="our_model")
    print("\nModel constructed")
    print(our_model.summary())
    return our_model


def small_model(input_size, num_features, num_regions, cell_num):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    resnet_output = resnet(x, input_size, 200)
    our_resnet = Model(inputs, resnet_output, name="our_resnet")
    num_patches = 250
    num_filters = 1006

    head_input = Input(shape=(num_patches, num_filters))
    x = head_input

    # x = tf.transpose(x, [0, 2, 1])
    # # x = Conv1D(num_regions, kernel_size=1, strides=1, use_bias=False, name="regions_projection")(x)
    # x = Dense(input_size // 100, activation=tf.nn.gelu, name="regions_projection")(x)
    # x = tf.transpose(x, [0, 2, 1])

    trim = (x.shape[-2] - num_regions) // 2
    x = x[..., trim + 1:-trim, :]

    x = Dropout(dropout_rate, input_shape=(num_regions, num_filters))(x)

    x = Conv1D(2048, kernel_size=1, strides=1, name="pointwise", activation=LeakyReLU(alpha=leaky_alpha))(x)
    outputs = Conv1D(cell_num, kernel_size=1, strides=1, name="last_conv1d")(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    print(outputs)
    head_output = outputs
    our_head = Model(head_input, head_output, name="our_head")
    print(our_head)

    our_model = Model(inputs, our_head(our_resnet(inputs)), name="our_model")
    print("\nModel constructed")
    print(our_model.summary())
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
    num_filters = 512

    # First convolutional layer. Since it is first, it is not preceded by activation and batch normalization
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters,
                     activation=False,
                     strides=3,
                     kernel_size=15,
                     name="rl_1_")
    current_len = input_size // 3
    # Instantiate the stack of residual units
    num_blocks = 7
    for block in range(num_blocks):
        cname = "rl_" + str(block) + "_"
        strides = 1
        y = x
        if block != num_blocks - 1:
            num_filters = int(num_filters * 1.12)
        activation = True
        if block != 0:
            # Downsample
            y = LeakyReLU(alpha=leaky_alpha, name="dwn_" + str(block))(y)
            y = tf.transpose(y, [0, 2, 1])
            current_len = math.ceil(current_len / 2)
            if block == num_blocks - 1:
                current_len = input_size // bin_size
            # Replace by conv maybe
            y = Dense(current_len, activation=LeakyReLU(alpha=leaky_alpha),
                      name="regions_projection_" + str(block))(y)
            y = tf.transpose(y, [0, 2, 1])
            strides = 2
            activation = False

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
