# import keras
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention,\
                                    Add, Embedding, Layer, Reshape, Dropout,\
                                    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2
import tensorflow as tf
import common as cm
from sam import sam_train_step
import numpy as np

projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4


def simple_model(input_size, num_regions, cell_num):
    input_shape = (input_size, 5)
    inputs = Input(shape=input_shape)
    x = inputs
    # x = Dropout(0.2)(x)
    x = resnet_v2(x, 8, 5)
    num_patches = 391
    # x = Dropout(0.5)(x)
    # for i in range(10):
    #     prev = x
    #     x = Conv1D(128, kernel_size=3, dilation_rate=2 ** (i+1),
    #                name="dilatation_" + str(i+1), padding='same')(x)
    #     x = BatchNormalization(name="bn_dilation_" + str(i+1))(x)
    #     if i != 0:
    #         x = Add()([x, prev])
    #     x = LeakyReLU(alpha=0.1, name="act_dilation_" + str(i+1))(x)

    encoded_patches = PatchEncoder(num_patches, projection_dim)(x)

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_1")(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name="mha_" + str(i)
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_2")(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, name="mlp_" + str(i))
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6, name="ln_rep")(encoded_patches)

    target_length = num_regions
    trim = (x.shape[-2] - target_length) // 2
    x = x[..., trim:-trim, :]
    x = Conv1D(2048, kernel_size=1, strides=1, name="pointwise", activation=tf.nn.gelu)(x)
    outputs = Conv1D(cell_num, kernel_size=1, strides=1, name="last_conv1d")(x)
    # trailing_axes = [-1, -2]
    # leading = tf.range(tf.rank(x) - len(trailing_axes))
    # trailing = trailing_axes + tf.rank(x)
    # new_order = tf.concat([leading, trailing], axis=0)
    outputs = tf.transpose(outputs, [0, 2, 1])
    outputs = LeakyReLU(alpha=0.1, name="model_final_output", dtype='float32')(outputs)
    print(outputs)
    model = Model(inputs, outputs, name="model")
    print("\nModel constructed")
    return model


class SAMModel(tf.keras.Model):
    def train_step(self, data):
        return sam_train_step(self, data)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=7,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name="rl_"):

    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  # use_bias=False,
                  name=name+"conv1d"
                  # kernel_regularizer=l2(1e-6),
                  # activity_regularizer=l2(1e-6)
                  )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name=name+"bn")(x)
        if activation is not None:
            x = LeakyReLU(alpha=0.1, name=name+"act")(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name=name+"bn")(x)
        if activation is not None:
            x = LeakyReLU(alpha=0.1, name=name+"act")(x)
        x = conv(x)
    return x


def resnet_v2(input_x, num_stages, num_res_blocks):
    # Start model definition.
    num_filters_in = 128

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters_in,
                     conv_first=True, name="rl_1_")

    # Instantiate the stack of residual units
    for stage in range(num_stages):
        for res_block in range(num_res_blocks):
            cname = "rl_" + str(stage) + "_" + str(res_block) + "_"
            activation = 'relu'
            batch_normalization = True
            strides = 1
            num_filters_out = int(num_filters_in * 1.1)
            if stage == 0:
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             name=cname+"1_")
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             name=cname+"2_")
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             name=cname+"3_")
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 name=cname+"4_")
            x = Add()([x, y])

        num_filters_in = num_filters_out

    x = BatchNormalization(name="res_bn_final")(x)
    x = LeakyReLU(alpha=0.1, name="res_act_final")(x)

    return x


def mlp(x, hidden_units, dropout_rate, name):
    for units in hidden_units:
        x = Dense(units, name=name + str(units), activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def wrap(input_sequences, output_scores, bs):
    train_data = tf.data.Dataset.from_tensor_slices((input_sequences, output_scores))
    train_data = train_data.batch(bs)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_data.with_options(options)
    return train_data


def wrap2(input_sequences, bs):
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
        self.projection = Dense(units=projection_dim, name="projection_dense")
        self.projection_dim = projection_dim
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim, name="pos_embedding"
        )

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