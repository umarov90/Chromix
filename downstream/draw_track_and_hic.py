import gc
import os
import pathlib
import tensorflow as tf
import joblib
from main_params import MainParams
import visualization as viz
import model as mo
import numpy as np

eval_gt_full = []
p = MainParams()
w_step = 500
predict_batch_size = 4
script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])
parsed_tracks_folder = folders[1]
parsed_hic_folder = folders[2]
model_folder = folders[3]
heads = joblib.load("pickle/heads.gz")
head_id = 0
head_tracks = heads[head_id]
p.parsed_hic_folder = folders[2]
hic_keys = joblib.load("pickle/hic_keys.gz")
for h in hic_keys:
    print(h)
infos = joblib.load("pickle/test_info.gz")[:100]
one_hot = joblib.load("pickle/one_hot.gz")
# hic_keys = [hic_keys[0]]
eval_track_names = []

for track in head_tracks:
    if "scEnd5" in track:
        eval_track_names.append(track)
        if len(eval_track_names) > 9:
            break

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = tf.keras.models.load_model(model_folder + p.model_name)
    our_model.get_layer("our_head").set_weights(joblib.load(model_folder + p.model_name + "_head_" + str(head_id)))

for i in range(len(infos)):
    eval_gt_full.append([])

for i, key in enumerate(eval_track_names):
    if i % 100 == 0:
        print(i, end=" ")
        gc.collect()
    parsed_track = joblib.load(p.parsed_tracks_folder + key)
    for j, info in enumerate(infos):
        start_bin = int(info[1] / p.bin_size) - p.half_num_regions
        extra_bin = start_bin + p.num_regions - len(parsed_track[info[0]])
        if start_bin < 0:
            binned_region = parsed_track[info[0]][0: start_bin + p.num_regions]
            binned_region = np.concatenate((np.zeros(-1 * start_bin), binned_region))
        elif extra_bin > 0:
            binned_region = parsed_track[info[0]][start_bin: len(parsed_track[info[0]])]
            binned_region = np.concatenate((binned_region, np.zeros(extra_bin)))
        else:
            binned_region = parsed_track[info[0]][start_bin:start_bin + p.num_regions]
        eval_gt_full[j].append(binned_region)

hic_output = []
for hi, key in enumerate(hic_keys):
    hdf = joblib.load(p.parsed_hic_folder + key)
    ni = 0
    for i, info in enumerate(infos):
        hd = hdf[info[0]]
        hic_mat = np.zeros((p.num_hic_bins, p.num_hic_bins))
        start_hic = int(info[1] - (info[1] % p.bin_size) - p.half_size_hic)
        end_hic = start_hic + 2 * p.half_size_hic
        start_row = hd['locus1_start'].searchsorted(start_hic - p.hic_bin_size, side='left')
        end_row = hd['locus1_start'].searchsorted(end_hic, side='right')
        hd = hd.iloc[start_row:end_row]
        # convert start of the input region to the bin number
        start_hic = int(start_hic / p.hic_bin_size)
        # subtract start bin from the binned entries in the range [start_row : end_row]
        l1 = (np.floor(hd["locus1_start"].values / p.hic_bin_size) - start_hic).astype(int)
        l2 = (np.floor(hd["locus2_start"].values / p.hic_bin_size) - start_hic).astype(int)
        hic_score = hd["score"].values
        # drop contacts with regions outside the [start_row : end_row] range
        lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
        l1 = l1[lix]
        l2 = l2[lix]
        hic_score = hic_score[lix]
        hic_mat[l1, l2] += hic_score
        hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
        if hi == 0:
            hic_output.append([])
        hic_output[ni].append(hic_mat)
        ni += 1
    del hd
    del hdf
    gc.collect()

test_seq = []
for info in infos:
    start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
    # if start in starts:
    #     continue
    # starts.append(start)
    extra = start + p.input_size - len(one_hot[info[0]])
    if start < 0:
        ns = one_hot[info[0]][0:start + p.input_size]
        ns = np.concatenate((np.zeros((-1 * start, p.num_features)), ns))
    elif extra > 0:
        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
        ns = np.concatenate((ns, np.zeros((extra, p.num_features))))
    else:
        ns = one_hot[info[0]][start:start + p.input_size]
    if len(ns) != p.input_size:
        print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
    test_seq.append(ns)

for w in range(0, len(test_seq), w_step):
    print(w, end=" ")
    p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
    if len(hic_keys) > 0:
        if w == 0:
            predictions_full = p1[0]
            predictions_hic = p1[1]
        else:
            predictions_full = np.concatenate((predictions_full, p1[0]))
            predictions_hic = np.concatenate((predictions_hic, p1[1]))

draw_arguments = [p, eval_track_names, predictions_full, eval_gt_full,
                  test_seq, infos, hic_keys,
                  predictions_hic, hic_output,
                  f"{p.figures_folder}/tracks/"]

joblib.dump(draw_arguments, "draw", compress=3)
# draw_arguments = joblib.load("draw")
viz.draw_tracks(*draw_arguments)
