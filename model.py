import torch
from torch.utils.data import Dataset
import numpy as np
from numba import jit
import math
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from main_params import MainParams 
from x_transformers import TransformerWrapper, Encoder


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# losses and metrics


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def nonzero_mse_loss(pred, target):
    mask = (target != 0).float()
    diff = pred - target
    masked_diff = mask * diff
    mse = torch.sum(masked_diff ** 2) / torch.sum(mask)
    return mse


def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

# main class

class Chromix(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.loss_weights = config.loss_weights
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, config.dim, num = 6, divisible_by = 2)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer
        # Bidirectional like BERT
        transformer = []
        for i in range(2):
            transformer.append(nn.Sequential(Encoder(
                dim = config.dim,
                depth = 2,
                heads = 4,
                rel_pos_max_distance = config.num_bins,
                rel_pos_num_buckets = 256,
                rel_pos_bias = True, 
                use_rmsnorm = True,
                ff_glu = True,
                ff_swish = True, # set this to True
                ff_no_bias = True,  # set this to True
                attn_talking_heads = True,  # turn on information exchange between attention heads
                attn_gate_values = True, # gate aggregated values with the input
                deepnorm = True, # set this to True to use deepnorm post-normalization configuration
                residual_attn = True,    # add residual attention
                layer_dropout = 0.1,   # stochastic depth - dropout entire layer
                attn_dropout = 0.1,    # dropout post-attention
                ff_dropout = 0.1       # feedforward dropout)
            )))

        self.transformer = nn.Sequential(*transformer)

        # target cropping

        self.target_length = config.num_bins
        self.crop_final = TargetLengthCrop(config.num_bins)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(0.05),
            GELU()
        )

        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        self.hic_projection = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(config.input_size // config.bin_size, config.hic_size),
            Rearrange('b d n -> b n d')
        )

        self._hic_heads = nn.ModuleDict()

        for specie in ["hg38", "mm10"]:
            hic_head = nn.Sequential(
                nn.Linear(self.dim * 2, len(config.hic_keys[specie])),
                nn.Softplus()
            )
            self._hic_heads[specie + "_hic"] = hic_head

        self.loss_weights = config.loss_weights

    def add_heads(self, **kwargs):
        self.output_heads = kwargs
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            self.crop_final,
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads
        

    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = None,
        target_length = None
    ):
        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        if exists(target_length):
            self.set_target_length(target_length)

        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.final_pointwise(x)

        if no_batch:
            x = rearrange(x, '() ... -> ...')

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        hx = self.hic_projection(x)
        out_hic = map_values(lambda fn: fn(hx), self._hic_heads)

        out.update(out_hic)

        # if exists(head):
        #     assert head in self._heads, f'head {head} not found'
        # out = out["expression"]

        if exists(target):
            assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

            if return_corr_coef:
                return pearson_corr_coef(out, target)

            loss = self.loss_weights["expression"] * poisson_loss(out["hg38_expression"], target["expression"])
            loss += self.loss_weights["epigenome"] * poisson_loss(out["hg38_epigenome"], target["epigenome"])
            loss += self.loss_weights["conservation"] * nn.MSELoss()(out["hg38_conservation"], target["conservation"])
            loss += self.loss_weights["hic"] * nn.MSELoss()(out["hg38_hic"], target["hic"])
            return loss

        if return_embeddings:
            return out, x

        return out


def batch_predict_effect(p, model, seqs1, seqs2, inds=None):
    for w in range(0, len(seqs1), p.w_step):
        print(w, end=" ")
        p1 = model.predict(wrap2(seqs1[w:w + p.w_step], p.predict_batch_size), verbose=0)
        pe1 = np.concatenate((p1[0], p1[1], p1[2]), axis=1)
        ph1 = p1[-1]
        if inds is not None:
            pe1 = pe1[:, inds, :]
        p2 = model.predict(wrap2(seqs2[w:w + p.w_step], p.predict_batch_size), verbose=0)
        pe2 = np.concatenate((p2[0], p2[1], p2[2]), axis=1)
        ph2 = p2[-1]
        if inds is not None:
            pe2 = pe2[:, inds, :]

        # effect_e = np.sum(pe1[..., p.mid_bin - 1 : p.mid_bin + 1], axis=-1) - np.sum(pe2[..., p.mid_bin - 1 : p.mid_bin + 1], axis=-1)
        # effect_e = np.mean(pe1 - pe2, axis=-1)
        # effect_e = np.max(np.abs(pe1 - pe2), axis=-1)
        effect_e = fast_ce(np.swapaxes(pe1, 1, 2), np.swapaxes(pe2, 1, 2))
        # effect_e = fast_ce(pe1, pe2)
        # effect_h = fast_ce(np.swapaxes(ph1, 1, 2), np.swapaxes(ph2, 1, 2))
        # effect_h = np.max(ph1 - ph2, axis=-1)
        effect_h = fast_ce(ph1, ph2)
        if w == 0:
            print(effect_e.shape)
            print(effect_h.shape)
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


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def cross_entropy(p, q):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    q = np.where(q > 1.0e-10, q, 1.0e-10)  # fill the zeros with 10**-10
    sl = [p[i] * np.log2(q[i]) for i in range(len(p))]
    sm = 0
    for a in sl:
        sm = sm + a
    return sm


# def JS_divergence(p,q):
#     M=(p+q)/2
#     return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)


# def KL_divergence(p,q):
#     return scipy.stats.entropy(p,q)


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def fast_ce(p1, p2):
    tmp1 = []
    for i in range(p1.shape[0]):
        tmp2 = []
        for j in range(p1.shape[1]):
            # tmp2.append(JS_divergence(normalization(p1[i][j]),normalization(p2[i][j])))
            # tmp2.append(scipy.stats.entropy(p1[i][j],p2[i][j],base=2))
            tmp2.append(cross_entropy(normalization(p1[i][j]), normalization(p2[i][j])))
        tmp1.append(tmp2)
    return np.array(tmp1)



class CustomDataset(Dataset):
    def __init__(self, data):
        self.inputs = []
        self.outputs = []
        for i in range(len(data[data.keys()[0]])):
            inputs_dict = {}
            outputs_dict = {}
            for key, val in data.items():  
                inputs_dict[key] = torch.from_numpy(val[0][i]).float()
                for key2, val2 in val[1].items():  
                    outputs_dict[key+"_"+key2] = torch.from_numpy(val2[i]).float()
            self.inputs.append(inputs_dict)
            self.outputs.append(outputs_dict)


    def __getitem__(self, index):
        input_data = self.inputs[index]
        output_data = self.outputs[index]
        return input_data, output_data

    def __len__(self):
        return len(self.inputs)


class DatasetDNA(Dataset):
    def __init__(self, inputs):
        self.inputs = torch.from_numpy(inputs).float()

    def __getitem__(self, index):
        input_data = self.inputs[index]
        return input_data

    def __len__(self):
        return len(self.inputs)
