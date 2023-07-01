import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numba import jit
import math
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange
from einops.layers.torch import Rearrange
from random import random
from sync_batchnorm import SynchronizedBatchNorm1d, convert_model


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

def get_groups_num(num_features):
    valid_num_groups = [
        factor for factor in range(1, num_features)
        if num_features % factor == 0
    ]
    infos = [
        {'ng': ng, 'nc': num_features / ng}
        for ng in valid_num_groups
    ]
    ideal = num_features ** (0.5)
    for item in infos:
        item['heuristic'] = abs(ideal - item['ng']) * abs(ideal - item['nc'])
    chosen = sorted(infos, key=lambda x: (x['heuristic'], 1 - x['ng']))[0]
    return chosen['ng']


class GaussianDropout(nn.Module):
    def __init__(self, p=0.01):
        super(GaussianDropout, self).__init__()
        self.alpha = (p / (1.0 - p)) ** 0.5

    def forward(self, x):
        if self.training:
            epsilon = torch.randn_like(x) * self.alpha + 1
            return x * epsilon
        else:
            return x


class Block(nn.Module):
    def __init__(self, dim, dr, layer_dropout=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=21, dilation=dr, padding='same', groups=dim)
        self.norm = nn.GroupNorm(get_groups_num(dim), dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        # self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_dropout = layer_dropout
        self.gaussian_dropout = GaussianDropout(0.01)

    def forward(self, x):
        if self.training and self.layer_dropout > 0.0 and random() < self.layer_dropout:
            return x
        input = x
        x = self.gaussian_dropout(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.pwconv1(x)
        x = self.dropout(x)
        x = self.act(x)
        # x = self.grn(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = rearrange(x, 'b n d -> b d n')
        if self.training and self.layer_dropout > 0.0:
            x = x / (1 - self.layer_dropout)
        x = input + x
        return x

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


def nonzero_mse_loss(pred, target):
    mask = (target != 0).float()
    diff = pred - target
    masked_diff = mask * diff
    loss1 = torch.mean(diff ** 2)
    loss2 = (torch.sum(masked_diff ** 2) / torch.sum(mask)) * 0.1
    if torch.isnan(loss2):
        loss2 = 0
    return loss1 + loss2

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        trim = (target_len - seq_len) // 2
        return x[:, -trim:trim]



class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out=None, kernel_size=1, dilation=1, use_act=True, gn=False):
        super().__init__()
        if gn:
            self.norm = nn.GroupNorm(get_groups_num(dim), dim)
        else:
            self.norm = nn.BatchNorm1d(dim, momentum=0.05)
        self.act = nn.GELU()
        self.use_act = use_act
        self.conv = nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding="same", dilation=dilation)

    def forward(self, x):
        x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        x = self.conv(x)
        return x

class Chromix(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        half_dim = config.dim // 2
        quad_dim = config.dim * 4
        # create stem
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            nn.MaxPool1d(kernel_size=2)
        )
        # create conv tower
        filter_list = exponential_linspace_int(half_dim, config.dim, num=6, divisible_by=128)
        filter_list = [half_dim, *filter_list]
        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                nn.MaxPool1d(kernel_size=2)
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        # Dilated variant
        # conv_layers = []
        # dr = 2
        # for i in range(20):
        #     conv_layers.append(Residual(nn.Sequential(
        #         ConvBlock(dim_out, dim_out, kernel_size=3, dilation=dr),
        #         ConvBlock(dim_out, dim_out, kernel_size=1),
        #         nn.Dropout(0.3)
        #     )
        #     ))
        #     dr = int(round(dr * 1.5))
        #
        # self.convnext = nn.Sequential(*conv_layers)

        num_blocks = 24
        dr = 1
        convnext_blocks = [nn.GroupNorm(get_groups_num(config.dim), config.dim)]
        layer_dropout = 0
        for block in range(num_blocks):
            print(dr)
            convnext_blocks.append(Block(dim=config.dim, dr=dr, layer_dropout=layer_dropout))
            dr = min(int(math.ceil(dr * 1.2)), (config.input_size // config.bin_size) // 10 - 10)
            layer_dropout = 0.15
        self.convnext = nn.Sequential(*convnext_blocks)
        self.target_length = config.num_bins
        self.crop_final = TargetLengthCrop(config.num_bins)
        # final pointwise
        self.final_pointwise = nn.Sequential(
            ConvBlock(filter_list[-1], quad_dim, 1, use_act=False, gn=True),
            Rearrange('b d n -> b n d'),
            nn.Dropout(0.05),
            nn.GELU()
        )
        # create final heads for human and mouse
        self.output_heads = config.output_heads
        self.heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            self.crop_final,
            nn.Linear(quad_dim, features)
        ), config.output_heads))

        self.hic_projection = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(config.input_size // config.bin_size, config.hic_size),
            Rearrange('b d n -> b n d')
        )

        self.hic_heads = nn.ModuleDict()

        for specie in ["hg38", "mm10"]:
            hic_head = nn.Sequential(
                nn.Linear(quad_dim, len(config.hic_keys[specie]))
            )
            self.hic_heads[specie + "_hic"] = hic_head

        self.running_loss = config.running_loss

    def forward(
        self,
        x,
        target = None,
        return_only_embeddings = False,
        head = None,
        loss_weights=None
    ):
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.convnext(x)
        # x = checkpoint_sequential(self.conv_tower, len(self.conv_tower), x)
        x = self.final_pointwise(x)

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self.heads)

        hx = self.hic_projection(x)
        out_hic = map_values(lambda fn: fn(hx), self.hic_heads)

        out.update(out_hic)

        if exists(target):
            losses = []
            for key in target.keys():
                t = target[key]
                subkey = key[len(head) + 1:]
                if subkey in ["conservation", "hic"]:
                    loss = nn.MSELoss()(out[key], t)
                else:
                    loss = nonzero_mse_loss(out[key], t)
                if not torch.isnan(loss):
                    self.running_loss[head].setdefault(subkey, []).append(loss.item())
                loss *= loss_weights[subkey]
                losses.append(loss)
            return sum(losses)

        return out


def batch_predict_effect(p, model, seqs1, seqs2, inds=None):
    model.eval()
    dd = DatasetDNA_ISM(seqs1, seqs2)
    ddl = DataLoader(dataset=dd, batch_size=p.pred_batch_size, shuffle=False)
    for batch, X in enumerate(ddl):
        print(batch, end=" ")
        X1 = X[0].to("cuda")
        X2 = X[1].to("cuda")
        with torch.no_grad():
            pr = model(X1)
            pe1 = np.concatenate((pr['hg38_expression'].cpu().numpy(),
                                 pr['hg38_epigenome'].cpu().numpy(),
                                 pr['hg38_conservation'].cpu().numpy()), axis=2)
            ph1 = pr['hg38_hic'].cpu().numpy()
            if inds is not None:
                pe1 = pe1[:, inds, :]

            pr = model(X2)
            pe2 = np.concatenate((pr['hg38_expression'].cpu().numpy(),
                                  pr['hg38_epigenome'].cpu().numpy(),
                                  pr['hg38_conservation'].cpu().numpy()), axis=2)
            ph2 = pr['hg38_hic'].cpu().numpy()
            if inds is not None:
                pe2 = pe2[:, inds, :]

        # effect_e = np.sum(pe1[..., p.mid_bin - 1 : p.mid_bin + 1], axis=-1) - np.sum(pe2[..., p.mid_bin - 1 : p.mid_bin + 1], axis=-1)
        # effect_e = np.mean(pe1 - pe2, axis=-1)
        # effect_e = np.max(np.abs(pe1 - pe2), axis=-1)
        # effect_e = fast_ce(np.swapaxes(pe1, 1, 2), np.swapaxes(pe2, 1, 2))
        effect_e = fast_ce(pe1, pe2)
        effect_h = fast_ce(np.swapaxes(ph1, 1, 2), np.swapaxes(ph2, 1, 2))
        # effect_h = np.max(ph1 - ph2, axis=-1)
        # effect_h = fast_ce(ph1, ph2)
        if batch == 0:
            print(effect_e.shape)
            print(effect_h.shape)
        fold_change = pe2[:, :, p.mid_bin] / pe1[:, :, p.mid_bin]
        # fold_change = np.squeeze(fold_change)
        # fold_change = np.max(fold_change, axis=1, keepdims=True)
        if batch == 0:
            effects_e = effect_e
            effects_h = effect_h
            fold_changes = fold_change
        else:
            effects_e = np.concatenate((effects_e, effect_e))
            effects_h = np.concatenate((effects_h, effect_h))
            fold_changes = np.concatenate((fold_changes, fold_change))
    print("")
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


def set_bn_train_mode(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.train()
            m.track_running_stats = True

def reset_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, SynchronizedBatchNorm1d):
            print("Resetting BN")
            module.reset_running_stats()


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
        for i in range(len(data[list(data.keys())[0]][0])):
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

class DatasetDNA_ISM(Dataset):
    def __init__(self, inputs1, inputs2):
        self.inputs1 = torch.from_numpy(inputs1).float()
        self.inputs2 = torch.from_numpy(inputs2).float()

    def __getitem__(self, index):
        input_data = (self.inputs1[index], self.inputs2[index])
        return input_data

    def __len__(self):
        return len(self.inputs1)

def load_weights(p, model, optimizer=None):
    if os.path.exists(p.model_folder + p.model_name):
        checkpoint = torch.load(p.model_folder + p.model_name, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        if exists(optimizer) and 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # optimizer.param_groups[2]['weight_decay'] = 1e-06
            # optimizer.param_groups[3]['weight_decay'] = 1e-06
            # for i, param_group in enumerate(optimizer.param_groups):
                # print(param_group)
            #    param_group['lr'] = p.lr # param_group['lr'] * 0.8
        del checkpoint
        torch.cuda.empty_cache()
        return epoch
    else:
        return 0


def prepare_model(p):
    model = Chromix(p)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    params = [
        {'params': model.stem.parameters()},
        {'params': model.conv_tower.parameters()},
        {'params': model.convnext.parameters(), 'weight_decay': 1e-06},
        {'params': model.final_pointwise.parameters()},
        {'params': model.heads.parameters()},
        {'params': model.hic_projection.parameters()},
        {'params': model.hic_heads.parameters()}
    ]
    optimizer = torch.optim.AdamW(params, lr=1e-04, weight_decay=0)
    model = nn.DataParallel(model)
    model = convert_model(model).to("cuda:0")
    return model, optimizer


def prepare_model_sc(p):
    model = Chromix_sc(p)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    params = [
        {'params': model.stem.parameters()},
        {'params': model.conv_tower.parameters()},
        {'params': model.convnext.parameters(), 'weight_decay': 1e-06},
        {'params': model.final_pointwise.parameters()},
        {'params': model.head.parameters()},
    ]
    optimizer = torch.optim.AdamW(params, lr=1e-06, weight_decay=0)
    model = nn.DataParallel(model)
    model = convert_model(model).to("cuda:0")
    return model, optimizer


def individual_clip(params):
    for p in params:
        torch.nn.utils.clip_grad_norm_(p, 0.01)