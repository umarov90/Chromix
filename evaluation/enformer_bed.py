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

params = MainParams()

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = f'{params.data_folder}data/species/hg38/genome.fa'
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
        center = interval.center() - interval.start2

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
        ax.fill_between(np.linspace(interval.start2, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    plt.savefig("enf_test.png")
    plt.close(fig)


model = Enformer(model_path)

fasta_extractor = FastaStringExtractor(fasta_file)
p = MainParams()
# tracks = {'DNASE:CD14-positive monocyte female': predictions[:, 41],
#           'DNASE:keratinocyte female': predictions[:, 42],
#           'CHIP:H3K27ac:keratinocyte female': predictions[:, 706],
#           'CAGE:Keratinocyte - epidermal': np.log10(1 + predictions[:, 4799])}
# plot_tracks(tracks, target_interval)
track_inds_bed = []
enf_tracks_id = {}
for index, row in df_targets.iterrows():
    if "monocytes" in row["description"].lower() or "t cells" in row["description"].lower():
        track_inds_bed.append(row["index"])
        enf_tracks_id[row["index"]] = row["identifier"]
        print(row["description"] + " " + row["identifier"])

print("Predicting")
num_bins = 896
bin_size = 128
all_start_vals = {}
for chrom in fasta_extractor._chromosome_sizes.keys():
    if chrom != "chr5":
        continue
    start_val = {}
    batch = []
    starts_for_bins = []
    for expression_region in range(0, fasta_extractor._chromosome_sizes[chrom], bin_size * num_bins):
        start = expression_region - SEQUENCE_LENGTH // 2 + (num_bins * bin_size) // 2
        target_interval = kipoiseq.Interval(chrom, start, start + SEQUENCE_LENGTH)
        sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval))
        batch.append(sequence_one_hot)
        starts_for_bins.append(expression_region)
        if len(batch) > 6 or expression_region + bin_size * num_bins > fasta_extractor._chromosome_sizes[chrom]:
            print(expression_region, end=" ")
            predictions = model.predict_on_batch(batch)['human']
            for c, locus in enumerate(predictions):
                start1 = starts_for_bins[c]
                for b in range(num_bins):
                    start2 = start1 + b * bin_size
                    for t in track_inds_bed:
                        track = enf_tracks_id[t]
                        start_val.setdefault(track, {})[start2] = locus[b][t]
            starts_for_bins = []
            batch = []
    print("")
    all_start_vals[chrom] = start_val


print("Saving bed files")
for track in start_val.keys():
    with open("bed_output/enf_prediction_" + track + ".bedGraph", 'w+') as f:
        for chrom in all_start_vals.keys():
            for start2 in sorted(all_start_vals[chrom][track].keys()):
                f.write(f"{chrom}\t{start2}\t{start2 + bin_size}\t{all_start_vals[chrom][track][start2]}")
                f.write("\n")