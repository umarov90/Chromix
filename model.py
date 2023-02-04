import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MaxPooling1D, BatchNormalization, \
    Concatenate, GaussianDropout, GaussianNoise, Add, Embedding, Layer, Dropout, Reshape, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization, LocallyConnected1D, DepthwiseConv1D, DepthwiseConv2D
from tensorflow.keras.models import Model
import numpy as np
from numba import jit


def make_model(input_size, num_features, num_regions, hic_num, hic_size, one_d_heads):
    inputs = Input(shape=(input_size, num_features))
    x = inputs
    output1d = body_d(x, input_size)
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

        hx = Activation(tf.keras.activations.softplus, dtype='float32')(hx)
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
    head_output = Activation(tf.keras.activations.softplus, dtype='float32')(outputs)
    return Model(head_input, head_output, name=name)


def conv_block(ix, filters, width, r, dr=1, b=True, a=True):
    x = ix
    if b:
        x = BatchNormalization()(x)
    if a:
        x = Activation(tf.nn.gelu)(x)
    x = Conv1D(filters, kernel_size=width, dilation_rate=dr, padding="same")(x)
    if r:
        x = x + ix
    return x


def body_d(input_x, input_size):
    num_filters = 512 + 256
    filter_nums = exponential_linspace_int(num_filters, 2 * num_filters, 6, divisible_by=128)
    x = Conv1D(num_filters,strides=1,kernel_size=15, padding="same")(input_x)
    x = conv_block(x, num_filters, 1, True)
    x = MaxPooling1D()(x)
    for block in range(6):
        num_filters = filter_nums[block]
        x = conv_block(x, num_filters, 5, False)
        x = conv_block(x, num_filters, 1, True)
        x = MaxPooling1D()(x)
    
    dr = 2
    for block in range(18):
        print(dr)
        y = conv_block(x, num_filters, 3, False, dr=dr)
        dr = int(round(dr * 1.5))
        y = conv_block(y, num_filters, 1, False)
        y = Dropout(0.3)(y)
        x = x + y
    
    x = conv_block(x, 2 * num_filters, 1, False)
    x = Dropout(0.05)(x)
    x = Activation(tf.nn.gelu, dtype='float32')(x)
    return x
    
    
def body_c(input_x, input_size):
    print("body c ver 1.05")
    num_filters = 1024
    filter_nums = exponential_linspace_int(num_filters, 2 * num_filters, 6, divisible_by=128)
    x = Conv1D(num_filters,strides=1,kernel_size=15, padding="same")(input_x)
    x = conv_block(x, num_filters, 1, True)
    x = MaxPooling1D()(x)
    for block in range(6):
        num_filters = filter_nums[block]
        x = conv_block(x, num_filters, 5, False)
        x = conv_block(x, num_filters, 1, True)
        x = MaxPooling1D()(x)
    
    current_len = input_size // 128
    num_block = 16
    for block in range(num_block):
        y = x
        y = gMLPLayer(current_len, num_filters, 0.1)(y)
        y = LayerScale(0.001, num_filters)(y)
        survival_probability = 1 - ( (block + 1) / num_block) * 0.5
        x = tfa.layers.StochasticDepth(survival_probability)([x, y])
    
    x = conv_block(x, 2 * num_filters, 1, False)
    x = Dropout(0.05)(x)
    x = Activation(tf.nn.gelu, dtype='float32')(x)
    return x
    
                
class LayerScale(layers.Layer):
    """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239.

    Args:
        init_values (float): value to initialize the diagonal matrix of LayerScale.
        projection_dim (int): projection dimension used in LayerScale.
    """

    def __init__(self, init_values: float, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(init_values * tf.ones((projection_dim,)))

    def call(self, x, training=False):
        return x * tf.cast(self.gamma, tf.float16) 
        
         
def body_cx(input_x, input_size):
    print("Version 1.44")
    # Initial number of filters
    num_filters = 1280 # 768 #
    mlp_start_block = 6
    num_blocks = 18
    patchify_val = 4
    # replace with hardcoded values
    filter_nums = exponential_linspace_int(num_filters, 2 * num_filters, mlp_start_block, divisible_by=128)
    # Patchify layer
    x = Conv1D(num_filters,strides=patchify_val,kernel_size=patchify_val,name="patchify")(input_x)
    x = LayerNormalization()(x)
    current_len = input_size // patchify_val
    for block in range(num_blocks):
        cname = "body_block_" + str(block) + "_"
        if block != 0 and block < mlp_start_block:
            # Downsample
            num_filters = filter_nums[block]
            strides = 2
            current_len = current_len // strides
            x = LayerNormalization()(x)
            x = Conv1D(num_filters, kernel_size=strides, strides=strides, padding="same", name=cname + "downsample")(x)
        y = x
        y = GaussianDropout(0.01)(y)
        # Spatial MLP for long range interactions
        if block >= mlp_start_block:
            z = y
            z = LayerNormalization()(z)
            z = tf.transpose(z, [0, 2, 1])
            z = Dense(5120, name=cname + "mlp_1")(z)
            z = Activation(tf.nn.gelu)(z)
            z = Dropout(0.1)(z)
            z = Dense(current_len, name=cname + "mlp_2")(z)
            z = tf.transpose(z, [0, 2, 1])
            z = LayerScale(0.001, num_filters)(z)
            y = z + y
        else:
            y = DepthwiseConv1D(kernel_size=7, name=cname + "depthwise", padding="same")(y)
        y = LayerNormalization()(y)
        # Pointwise to mix the channels
        y = Conv1D(2 * num_filters, kernel_size=1, padding="same", name=cname + "pointwise_1")(y)
        y = Activation(tf.nn.gelu)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(num_filters, kernel_size=1, padding="same", name=cname + "pointwise_2")(y)
        y = LayerScale(0.001, num_filters)(y)
        survival_probability = 1 - ( (block + 1) / num_blocks) * 0.5
        x = tfa.layers.StochasticDepth(survival_probability)([x, y])

    x = LayerNormalization()(x)
    x = Conv1D(2 * num_filters, kernel_size=1, name="body_output", activation=tf.nn.gelu, dtype='float32')(x)
    x = Dropout(0.1)(x)
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


def batch_predict_effect(p, model, seqs1, seqs2, inds=None):
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        p1 = model.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose = 0)
        pe1 = p1[0]
        ph1 = p1[1]
        if inds is not None:
            pe1 = pe1[:, inds, :]
        p2 = model.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose = 0)
        pe2 = p2[0]
        ph2 = p2[1]
        if inds is not None:
            pe2 = pe2[:, inds, :]

        effect_e = fast_ce(np.swapaxes(pe1, 1, 2), np.swapaxes(pe2, 1, 2))
        effect_h = fast_ce(np.swapaxes(ph1, 1, 2), np.swapaxes(ph2, 1, 2))
        fold_change = pe2[:, :, p.mid_bin] / pe1[:, :, p.mid_bin]
        # fold_change = np.squeeze(fold_change)
        # fold_change = np.max(fold_change, axis=1, keepdims=True)
        if w == 0:
            effects_e = effect_e
            effects_h = effect_h
            fold_changes = fold_change
        else:
            effects_e = np.concatenate((effects_e, effect_e))
            effects_h = np.concatenate((effects_h, effect_h))
            fold_changes = np.concatenate((fold_changes, fold_change))
    fold_changes = np.clip(fold_changes, 0, 100)
    fold_changes = np.log(fold_changes + 1)
    fold_changes[np.isnan(fold_changes)] = -1
    return effects_e, effects_h, fold_changes

    
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def cross_entropy(p, q):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    q = np.where(q>1.0e-10,q,1.0e-10) #fill the zeros with 10**-10
    sl = [p[i]*np.log2(q[i]) for i in range(len(p))]
    sm = 0
    for a in sl:
        sm = sm + a
    return sm

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

        effect = fast_ce(p1, p2)
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