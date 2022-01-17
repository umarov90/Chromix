# import keras
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention, \
    Add, Embedding, Layer, Reshape, Dropout, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2
import tensorflow as tf
import common as cm
import numpy as np

projection_dim = 128
# dropout_rate = 0.2
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8


def hic_model(input_size, num_features, num_regions, cell_num, hic_num, hic_size):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    x = Dropout(0.5, input_shape=input_shape)(x)
    resnet_output = resnet(x, 8, 2)
    our_resnet = Model(inputs, resnet_output, name="our_resnet")
    num_patches = 1641  # 391 #
    num_filters = 408
    interactions_layer_input = Input(shape=(num_patches, num_filters))
    interactions_layer_input = Dropout(0.5, input_shape=(num_patches, num_filters))(interactions_layer_input)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(interactions_layer_input)

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_1")(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, name="mha_" + str(i)
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_2")(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, name="mlp_" + str(i))
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6, name="ln_rep")(encoded_patches)
    interactions_layer_output = x

    our_interactions_layer = Model(interactions_layer_input, interactions_layer_output, name="our_transformer")

    hic_input = Input(shape=(num_patches, projection_dim))
    hx = Conv1D(32, kernel_size=1, strides=1, name="pointwise_hic", activation=tf.nn.gelu)(hic_input)
    hx = Flatten()(hx)
    h_layers = []
    for h in range(hic_num):
        h_layers.append(Dense(hic_size, activation=tf.keras.activations.softplus)(hx))
    hic_output = tf.stack(h_layers, axis=1)
    print(hic_output)
    hic_act = LeakyReLU(alpha=0.1, name="hic_output", dtype='float32')(hic_output)
    our_hic = Model(hic_input, hic_act, name="our_hic")

    head_input = Input(shape=(num_patches, projection_dim))
    x = head_input
    x = Dropout(0.5, input_shape=(num_patches, projection_dim))(x)
    target_length = num_regions
    trim = (x.shape[-2] - target_length) // 2
    x = x[..., trim:-trim, :]

    x = Conv1D(2048, kernel_size=1, strides=1, name="pointwise", activation=tf.nn.gelu)(x)
    outputs = Conv1D(cell_num, kernel_size=1, strides=1, name="last_conv1d", dtype='float32')(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    print(outputs)
    head_output = LeakyReLU(alpha=0.1, name="model_final_output", dtype='float32')(outputs)
    our_head = Model(head_input, head_output, name="our_head")
    print(our_head)

    our_model = Model(inputs, [our_head(our_interactions_layer(our_resnet(inputs))),
                               our_hic(our_interactions_layer(our_resnet(inputs)))], name="our_model")
    print("\nModel constructed")
    print(our_model.summary())
    return our_model


def small_model(input_size, num_features, num_regions, cell_num):
    input_shape = (input_size, num_features)
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    # x = Dropout(dropout_rate, input_shape=input_shape)(x)
    resnet_output = resnet(x, 8, 1)
    our_resnet = Model(inputs, resnet_output, name="our_resnet")
    num_patches = 469
    num_filters = 1093

    interactions_layer_input = Input(shape=(num_patches, num_filters))
    encoded_patches = PatchEncoder(num_patches, projection_dim)(interactions_layer_input)
    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_1")(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, name="mha_" + str(i)
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_2")(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, name="mlp_" + str(i))
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6, name="ln_rep")(encoded_patches)
    interactions_layer_output = x

    our_interactions_layer = Model(interactions_layer_input, interactions_layer_output, name="our_transformer")

    head_input = Input(shape=(num_patches, projection_dim))
    x = head_input

    x = tf.transpose(x, [0, 2, 1])
    x = Conv1D(num_regions, kernel_size=1, strides=1, use_bias=False, name="regions_projection")(x)
    x = tf.transpose(x, [0, 2, 1])

    # x = Dropout(dropout_rate, input_shape=(num_regions, projection_dim))(x)

    x = Conv1D(2048, kernel_size=1, strides=1, name="pointwise", activation=tf.nn.gelu)(x)
    outputs = Conv1D(cell_num, kernel_size=1, strides=1, name="last_conv1d")(x)
    outputs = tf.transpose(outputs, [0, 2, 1])
    print(outputs)
    head_output = LeakyReLU(alpha=0.1, name="model_final_output", dtype='float32')(outputs)
    our_head = Model(head_input, head_output, name="our_head")
    print(our_head)

    our_model = Model(inputs, our_head(our_interactions_layer(our_resnet(inputs))), name="our_model")
    print("\nModel constructed")
    print(our_model.summary())
    return our_model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation=True,
                 batch_normalization=True,
                 name="rl_"):
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name=name + "conv1d"
                  # kernel_regularizer=l2(1e-6),
                  # activity_regularizer=l2(1e-6)
                  )

    x = inputs
    if activation:
        x = LeakyReLU(alpha=0.1, name=name + "act")(x)
    if batch_normalization:
        x = BatchNormalization(name=name + "bn")(x)
    x = conv(x)
    return x


def resnet(input_x, num_stages, num_res_blocks):
    # Start model definition.
    num_filters = 512

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters,
                     activation=False,
                     batch_normalization=False,
                     name="rl_1_")

    # Instantiate the stack of residual units
    for stage in range(num_stages):
        for res_block in range(num_res_blocks):
            cname = "rl_" + str(stage) + "_" + str(res_block) + "_"
            strides = 1
            num_filters = int(num_filters * 1.1)
            if res_block == 0 and stage != 0:
                strides = 2

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             name=cname + "1_")

            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             name=cname + "2_")
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=False,
                                 batch_normalization=False,
                                 name=cname + "3_")
            x = Add()([x, y])

    x = LeakyReLU(alpha=0.1, name="res_act_final")(x)
    x = BatchNormalization(name="res_bn_final")(x)

    return x


def mlp(x, hidden_units, name): # dropout_rate,
    for units in hidden_units:
        x = Conv1D(units,
                   kernel_size=1,
                   strides=1,
                   activation=tf.nn.gelu,
                   name=name + str(units))(x)
        # x = Dropout(dropout_rate)(x)
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


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Conv1D(projection_dim, kernel_size=1, strides=1, use_bias=False, name="projection_patch_encoder")
        self.projection_dim = projection_dim
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim, name="pos_embedding")

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config
