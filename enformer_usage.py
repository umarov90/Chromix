import tensorflow as tf
import numpy as np
from numba import jit

SEQUENCE_LENGTH = 393216


class Enformer:

    def __init__(self):
        self._model = tf.saved_model.load("data/enformer_model")

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}


class EnformerScoreVariantsRaw:

    def __init__(self, organism='human'):
        self._model = Enformer()
        self._organism = organism

    def predict_on_batch(self, inputs, inds=None):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]
        # effect = alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)
        # effect = fast_ce(alt_prediction, ref_prediction)
        effect = fast_ce(np.swapaxes(alt_prediction, 1, 2), np.swapaxes(ref_prediction, 1, 2))
        # effect = np.max(np.abs(alt_prediction - ref_prediction), axis=1)
        # effect = alt_prediction[:, 447:450, :].sum(axis=1) - ref_prediction[:, 447:450, :].sum(axis=1)
        if inds is None:
            r = np.squeeze(ref_prediction[:, 447:450, :].sum(axis=1))
            f = np.squeeze(alt_prediction[:, 447:450, :].sum(axis=1))
        else:
            r = np.squeeze(ref_prediction[:, 447:450, inds].sum(axis=1))
            f = np.squeeze(alt_prediction[:, 447:450, inds].sum(axis=1))
        fold_change = np.abs(f - r).max()
        print(fold_change)
        return effect[0], fold_change


enformer_score_variants = EnformerScoreVariantsRaw()


def calculate_effect(seqs1, seqs2, inds=None):
    if inds is not None and len(inds)==0:
        inds=None
    effects = []
    fold_changes = []
    for i in range(len(seqs1)):
        if i % 100 == 0:
            print(i, end=" ")
        variant_scores, fold_change = enformer_score_variants.predict_on_batch({"ref": seqs1[i][np.newaxis], "alt": seqs2[i][np.newaxis]}, inds)
        effects.append(variant_scores)
        fold_changes.append(fold_change)
    fold_changes = np.asarray(fold_changes)
    # fold_changes = np.clip(fold_changes, 0, 100)
    # fold_changes = np.log(fold_changes + 1)
    # fold_changes[np.isnan(fold_changes)] = -1
    return np.asarray(effects), fold_changes


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
