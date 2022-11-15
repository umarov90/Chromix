import model as mo
from main_params import MainParams
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import parse_data as parser
from pathlib import Path

compute_correlation = False
p = MainParams()
head = joblib.load(f"{p.pickle_folder}heads.gz")
head = head["sc"]
track_inds_bed = []

# cor_tracks = pd.read_csv("data/fantom_tracks.tsv", sep="\t", header=None).iloc[:, 0].tolist()

for i, track in enumerate(head):
    # if track in cor_tracks:
    track_inds_bed.append(i)

print(f"Predicting {len(track_inds_bed)} tracks")

one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head)
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_sc"))

all_start_vals = {}
corrs = []
for chrom in ["chr21"]: # one_hot.keys()
    print(f"\nPredicting {chrom} +++++++++++++++++++++++")
    start_val = {}
    batch = []
    starts_for_bins = []
    output_scores_info = []
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
        start_bin = expression_region // p.bin_size
        output_scores_info.append([chrom, start_bin, start_bin + p.num_bins])
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

            if compute_correlation:
                output_expression = parser.par_load_data(output_scores_info, head, p)

                x = pred
                y = output_expression

                x_m = x - np.mean(x, axis=1)[:, None]
                y_m = y - np.mean(y, axis=1)[:, None]

                X = np.sum(x_m ** 2, axis=1)
                Y = np.sum(y_m ** 2, axis=1)

                corr = np.dot(x_m, y_m.T) / np.sqrt(np.dot(X[:, None], Y[None]))

            output_scores_info = []
            starts_for_bins = []
            batch = []
    all_start_vals[chrom] = start_val

if compute_correlation:
    print("Average correlation: " + str(np.mean([corrs])))

print("Saving bed files")
Path("bed_output").mkdir(parents=True, exist_ok=True)
for track in start_val.keys():
    with open("bed_output/our_" + track + ".bedGraph", 'w+') as f:
        for chrom in all_start_vals.keys():
            for start2 in sorted(all_start_vals[chrom][track].keys()):
                f.write(f"{chrom}\t{start2}\t{start2 + p.bin_size}\t{all_start_vals[chrom][track][start2]}")
                f.write("\n")
