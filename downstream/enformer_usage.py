import gc
import os
import pickle
import multiprocessing as mp
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
from scipy import stats
import pathlib
import pyBigWig
import model as mo
from main_params import MainParams

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = '/media/user/passport1/variants_100/data/hg38.fa'
clinvar_vcf = '/root/data/clinvar.vcf.gz'

# Download targets from Basenji2 dataset
# Cite: Kelley et al Cross-species regulatory sequence activity prediction. PLoS Comput. Biol. 16, e1008050 (2020).
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')

pyfaidx.Faidx(fasta_file)

SEQUENCE_LENGTH = 393216


class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

    def __init__(self, tfhub_url, organism='human'):
        self._model = Enformer(tfhub_url)
        self._organism = organism

    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                 organism='human'):
        assert organism == 'human', 'Transforms only compatible with organism=human'
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            transform_pipeline = joblib.load(f)
        self._transform = transform_pipeline.steps[0][1]  # StandardScaler.

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                 organism='human', num_top_features=500):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            self._transform = joblib.load(f)
        self._num_top_features = num_top_features

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)[:, :self._num_top_features]


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
    """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""

    def _open(file):
        return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)

    with _open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, id, ref, alt_list = line.split('\t')[:5]
            # Split ALT alleles and return individual variants as output.
            for alt in alt_list.split(','):
                yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                                   ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
        reference_sequence=FastaStringExtractor(fasta_file))

    for variant in variant_generator(vcf_file, gzipped=gzipped):
        interval = Interval(chr_prefix + variant.chrom,
                            variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start

        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)

        yield {'inputs': {'ref': one_hot_encode(reference),
                          'alt': one_hot_encode(alternate)},
               'metadata': {'chrom': chr_prefix + variant.chrom,
                            'pos': variant.pos,
                            'id': variant.id,
                            'ref': variant.ref,
                            'alt': variant.alt}}


def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    plt.savefig("enf_test.png")
    plt.close(fig)


model = Enformer(model_path)

fasta_extractor = FastaStringExtractor(fasta_file)

script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])
parsed_tracks_folder = folders[1]
parsed_hic_folder = folders[2]
model_folder = folders[3]
params = MainParams()
# tracks = {'DNASE:CD14-positive monocyte female': predictions[:, 41],
#           'DNASE:keratinocyte female': predictions[:, 42],
#           'CHIP:H3K27ac:keratinocyte female': predictions[:, 706],
#           'CAGE:Keratinocyte - epidermal': np.log10(1 + predictions[:, 4799])}
# plot_tracks(tracks, target_interval)

# enformer_test_tss TRY !!!!!!!!!!!!!!!!!!!!!
gene_tss = pd.read_csv("data/enformer/enformer_test_tss_less.bed", sep="\t", index_col=False,
                       names=["chr", "start", "end", "type"])
# gene_tss = gene_tss.loc[gene_tss['chr'] == "chr1"]
# gene_tss.index = range(len(gene_tss.index))
# gene_tss = gene_tss.head(100)
# eval_tracks = pd.read_csv("data/eval_tracks.tsv", sep=",", header=None)[0].tolist()
enf_tracks = df_targets[df_targets['description'].str.contains("CAGE")]['identifier'].tolist()
# extract from enformer targets identifier where row contains cage
# full_name = pickle.load(open(f"pickle/full_name.p", "rb"))
heads = joblib.load("pickle/heads.gz")
head_id = 0
head_tracks = heads[head_id]
track_ind_our = {}
eval_tracks = []
for track in enf_tracks:
    for our_track in head_tracks:
        if track in our_track:
            track_ind_our[track] = head_tracks.index(our_track)
            eval_tracks.append(track)
            break

full_name = {}
tracks_folder = "/media/user/EE3C38483C380DD9/bw/"
for filename in os.listdir(tracks_folder):
    for track in eval_tracks:
        if track in filename:
            size = os.path.getsize(tracks_folder + filename)
            full_name[track] = filename
# pickle.dump(full_name, open(f"pickle/full_name.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
# np.savetxt("data/eval_tracks.tsv", eval_tracks, delimiter="\t", fmt='% s')
print(f"Eval tracks {len(eval_tracks)}")
print(f"TSS {len(gene_tss)}")
# track_ind = pickle.load(open(f"pickle/track_ind.p", "rb"))
track_ind = {}
for j, track in enumerate(eval_tracks):
    track_ind[track] = df_targets[df_targets['identifier'].str.contains(track)].index

# pickle.dump(track_ind, open(f"pickle/track_ind.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# ENFORMER #####################################################################################################
#################################################################################################################

pred_matrix = joblib.load("/media/user/EE3C38483C380DD9/temp/pred_matrix.p")
# pred_matrix = np.zeros((len(eval_tracks), len(gene_tss)))
# counts = [0, 0, 0]
# batch = []
# print("Predicting")
# gene_index = 0
# for index, row in gene_tss.iterrows():
#     if index % 100 == 0:
#         gc.collect()
#         print(index, end=" ")
#     target_interval = kipoiseq.Interval(row["chr"], row["start"] - 63, row["start"] - 63 + 1)
#     sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
#     batch.append(sequence_one_hot)
#     if len(batch) > 6 or index == len(gene_tss) - 1:
#         predictions = model.predict_on_batch(batch)['human']
#         for p in predictions:
#             for j, track in enumerate(eval_tracks):
#                 t = track_ind[track]
#                 bins = [p[447, t], p[448, t], p[449, t]]
#                 # bins = [p[448, t]]
#                 gene_expression = np.sum(bins)
#                 counts[bins.index(max(bins))] += 1
#                 # np.asarray(p)[:, 4799]
#                 pred_matrix[j, gene_index] = gene_expression
#             gene_index += 1
#         batch = []
#
# print("")
# print(counts)
# joblib.dump(pred_matrix, "/media/user/EE3C38483C380DD9/temp/pred_matrix.p", compress=3)

# OUR MODEL #####################################################################################################
#################################################################################################################
one_hot = joblib.load("pickle/one_hot.gz")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = tf.keras.models.load_model(model_folder + params.model_name)
    our_model.get_layer("our_head").set_weights(joblib.load(model_folder + params.model_name + "_head_" + str(head_id)))

pred_matrix_our = np.zeros((len(eval_tracks), len(gene_tss)))
p = MainParams()
test_seq = []
predict_batch_size = p.GLOBAL_BATCH_SIZE
w_step = 500
counts = [0, 0, 0]
print("Predicting our")
gene_index = 0
for index, row in gene_tss.iterrows():
    start = int(row["start"] - p.half_size)
    ns = one_hot[row["chr"]][start:start + p.input_size]
    test_seq.append(ns)

for w in range(0, len(test_seq), w_step):
    print(w, end=" ")
    p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
    for m in range(len(p1)):
        for j, track in enumerate(eval_tracks):
            t = track_ind_our[track]
            bins = [p1[m, t, p.mid_bin - 1], p1[m, t, p.mid_bin], p1[m, t, p.mid_bin + 1]]
            counts[bins.index(max(bins))] += 1
    p2 = p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
    if w == 0:
        predictions = p2
    else:
        predictions = np.concatenate((predictions, p2), dtype=np.float32)

for i in range(len(gene_tss)):
    for j, track in enumerate(eval_tracks):
        t = track_ind_our[track]
        pred_matrix_our[j, i] = predictions[i, t]

print("")
print(counts)

# GT DATA #####################################################################################################
#################################################################################################################

gt_matrix = np.zeros((len(eval_tracks), len(gene_tss)))

bin = 128 # 128
half_bin = 63 # 63

# for track in eval_tracks:
#     gt_vector = np.zeros(len(gene_tss))
#     bw = pyBigWig.open(tracks_folder + full_name[track])
#     for index, row in gene_tss.iterrows():
#         start = row["start"] - half_bin - bin
#         vals = bw.values(row["chr"], start, start + 3 * bin)
#         vals = np.nan_to_num(vals)
#         vals = [np.sum(vals[0:bin]), np.sum(vals[bin:2 * bin]), np.sum(vals[2 * bin:])]
#         gt_count = 0
#         for v in vals:
#             if v > 0:
#                 gt_count += np.log10(v)
#         gt_vector[index] = gt_count
#     bw.close()
#     joblib.dump(gt_vector, "/media/user/EE3C38483C380DD9/temp/gt_" + str(eval_tracks.index(track)))
#     print(track)


def load_bw(q, sub_tracks):
    for track in sub_tracks:
        gt_vector = np.zeros(len(gene_tss))
        bw = pyBigWig.open(tracks_folder + full_name[track])
        for index, row in gene_tss.iterrows():
            start = row["start"] - half_bin - bin
            vals = bw.values(row["chr"], start, start + 3 * bin)
            vals = np.nan_to_num(vals)
            vals = np.asarray([np.sum(vals[0:bin]), np.sum(vals[bin:2 * bin]), np.sum(vals[2 * bin:])])
            gt_vector[index] = np.sum(vals)

            # start = row["start"] - 64
            # vals = bw.values(row["chr"], start, start + 128)
            # vals = np.nan_to_num(vals)
            # gt_vector[index] = np.sum(vals)
        bw.close()
        joblib.dump(gt_vector, "/media/user/EE3C38483C380DD9/temp/gt_"+str(eval_tracks.index(track)))
        print(track)
    q.put(None)


print("Loading GT")
step_size = 30
q = mp.Queue()
ps = []
start = 0
nproc = 10
end = len(eval_tracks)
for t in range(start, end, step_size):
    t_end = min(t+step_size, end)
    sub_tracks = eval_tracks[t:t_end]
    p = mp.Process(target=load_bw,
                   args=(q, sub_tracks,))
    p.start()
    ps.append(p)
    if len(ps) >= nproc:
        for p in ps:
            p.join()
        print(q.get())
        ps = []

if len(ps) > 0:
    for p in ps:
        p.join()
    print(q.get())

print("")
for i in range(len(eval_tracks)):
    gt_matrix[i, :] = joblib.load("/media/user/EE3C38483C380DD9/temp/gt_"+str(i))

print("Enformer =================================================")
corrs = []
for i in range(len(eval_tracks)):
    a = pred_matrix[i, :]
    b = gt_matrix[i, :]
    corr = stats.spearmanr(a, b)[0]
    corrs.append(corr)

print(f"Across tracks {np.median(np.nan_to_num(corrs))}")

corrs = []
for i in range(len(gene_tss)):
    a = pred_matrix[:, i]
    b = gt_matrix[:, i]
    corr = stats.spearmanr(a, b)[0]
    corrs.append(corr)

print(f"Across genes {np.median(np.nan_to_num(corrs))}")

print("OUR =================================================")
corrs = []
for i in range(len(eval_tracks)):
    a = pred_matrix_our[i, :]
    b = gt_matrix[i, :]
    corr = stats.spearmanr(a, b)[0]
    corrs.append(corr)

print(f"Across tracks {np.median(np.nan_to_num(corrs))}")

corrs = []
for i in range(len(gene_tss)):
    a = pred_matrix_our[:, i]
    b = gt_matrix[:, i]
    corr = stats.spearmanr(a, b)[0]
    corrs.append(corr)

print(f"Across genes {np.median(np.nan_to_num(corrs))}")


