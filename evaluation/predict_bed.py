import model as mo
from main_params import MainParams
import joblib
import tensorflow as tf
import numpy as np


head_name = "hg38"

p = MainParams()
heads = joblib.load("pickle/heads.gz")
head = heads[head_name]
if head_name == "hg38":
    head = head["expression"]
track_inds_bed = []

with open('candidate_tracks.tsv') as f:
    candidates_list = f.read().splitlines()

for i, track in enumerate(head):
    if track in candidates_list:
        track_inds_bed.append(i)

one_hot = joblib.load(f"pickle/{head_name}_one_hot.gz")

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.small_model(p.input_size, p.num_features, p.num_bins, len(head), p.bin_size)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_" + head_name))

all_start_vals = {}
for chrom in ["chr11"]:#one_hot.keys():
    print(f"\nPredicting {chrom} +++++++++++++++++++++++")
    start_val = {}
    batch = []
    starts_for_bins = []
    for expression_region in range(0, len(one_hot[chrom]), p.bin_size * p.num_bins):
        start = expression_region - p.half_size + (p.num_bins * p.bin_size) // 2
        extra = start + p.input_size - len(one_hot[chrom])
        if start < 0:
            ns = one_hot[chrom][0:start + p.input_size]
            ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
        elif extra > 0:
            ns = one_hot[chrom][start: len(one_hot[chrom])]
            ns = np.concatenate((ns, np.zeros((extra, 5))))
        else:
            ns = one_hot[chrom][start:start + p.input_size]
        batch.append(ns[:, :-1])
        starts_for_bins.append(expression_region)
        if len(batch) > p.w_step or expression_region + p.bin_size * p.num_bins > len(one_hot[chrom]):
            print(expression_region, end=" ")
            batch = np.asarray(batch, dtype=bool)
            pred = our_model.predict(mo.wrap2(batch, p.predict_batch_size), batch_size=p.predict_batch_size)
            for c, locus in enumerate(pred):
                start1 = starts_for_bins[c]
                for b in range(p.num_bins):
                    start2 = start1 + b * p.bin_size
                    for t in track_inds_bed:
                        track = head[t]
                        start_val.setdefault(track, {})[start2] = locus[t][b]
            starts_for_bins = []
            batch = []
    all_start_vals[chrom] = start_val


print("Saving bed files")
for track in start_val.keys():
    with open("bed_output/our_" + track + ".bedGraph", 'w+') as f:
        for chrom in all_start_vals.keys():
            for start2 in sorted(all_start_vals[chrom][track].keys()):
                f.write(f"{chrom}\t{start2}\t{start2 + p.bin_size}\t{all_start_vals[chrom][track][start2]}")
                f.write("\n")
