import io
import os

import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gc
import random
import pandas as pd
import math
import time
import attribution
import numpy as np
import common as cm
from pathlib import Path
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from heapq import nsmallest
import copy
import seaborn as sns
from multiprocessing import Pool
import shutil
import psutil
import sys
import parse_data as parser
from datetime import datetime
import traceback
import multiprocessing as mp
import pathlib
import visualization as viz
import pickle
matplotlib.use("agg")
from scipy.ndimage.filters import gaussian_filter
from itertools import repeat
# tf.compat.v1.disable_eager_execution()
# import tensorflow as tf
import tensorflow_addons as tfa

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
input_size = 210001 # 50001
half_size = int(input_size / 2)
bin_size = 200
hic_bin_size = 10000
num_hic_bins = 20
half_size_hic = 100000
num_regions = 501  # 201
half_num_regions = int(num_regions / 2)
mid_bin = math.floor(num_regions / 2)
BATCH_SIZE = 2
GLOBAL_BATCH_SIZE = 4 * BATCH_SIZE # 3 ????
STEPS_PER_EPOCH = 200
num_epochs = 2000
hic_track_size = 1
out_stack_num = 4000
num_features = 5
shift_speed = 2000000
initial_shift = 0
last_proc = None
hic_size = 190
model_folder = "/home/user/data/models"
model_name = "all_tracks.h5" #"200k_192_1_5features.h5" "new_big.h5"
figures_folder = "figures_1"
tracks_folder = "/home/user/data/tracks/"
temp_folder = "/home/user/data/temp/"
parsed_tracks_folder = "/home/user/data/parsed_tracks/"
chromosomes = ["chrX"]  # "chrY"
for i in range(1, 23):
    chromosomes.append("chr" + str(i))
mixed16 = False


def recompile(q, lr, unfreeze=0):
    import tensorflow as tf
    import model as mo
    if mixed16:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + model_name,
                                               custom_objects={'PatchEncoder': mo.PatchEncoder})
        if unfreeze > 0:
            our_model.trainable = True

        print(datetime.now().strftime('[%H:%M:%S] ') + "Compiling model")
        loss_weights = {"our_head": 1.0, "our_hic": 0.15}
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, name="SGD") # momentum=0.9, nesterov=True,
        if len(hic_keys) > 0:
            our_model.compile(loss="mse", optimizer=optimizer, loss_weights=loss_weights)
        else:
            our_model.compile(loss="mse", optimizer=optimizer)

        Path(model_folder).mkdir(parents=True, exist_ok=True)
        our_model.save(model_folder + model_name)
        print("Model saved " + model_folder + model_name)
    q.put(None)


def create_model(q):
    import tensorflow as tf
    import model as mo
    if mixed16:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = mo.hic_model(input_size, num_features, num_regions, out_stack_num, hic_num, hic_size)
        print("loading model")
        our_model_old = tf.keras.models.load_model(model_folder + "new_big.h5_no.h5",
                                                   custom_objects={'PatchEncoder': mo.PatchEncoder})
        print("model loaded")
        for layer in our_model_old.get_layer("our_resnet").layers:
            layer_name = layer.name
            layer_weights = layer.weights
            our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(layer_weights)

        for layer in our_model_old.get_layer("our_transformer").layers:
            layer_name = layer.name
            layer_weights = layer.weights
            if "input" in layer_name:
                continue
            if isinstance(layer, mo.PatchEncoder):
                layer_weights = layer.projection.weights
                our_model.get_layer("our_transformer").get_layer(layer_name).projection.set_weights(layer_weights)
            elif isinstance(layer.weights, list):
                try:
                    new_weights_list = []
                    for li in range(len(layer.weights)):
                        new_shape = our_model.get_layer("our_transformer").get_layer(layer_name).weights[li].shape
                        new_weights_list.append(np.resize(layer.weights[li], new_shape))
                    our_model.get_layer("our_transformer").get_layer(layer_name).set_weights(new_weights_list)
                except:
                    pass
            else:
                new_shape = our_model.get_layer("our_transformer").get_layer(layer_name).shape
                our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(np.resize(layer_weights, new_shape))

        # for layer in our_model_old.get_layer("our_head").layers:
        #     layer_name = layer.name
        #     if "input" in layer_name:
        #         continue
        #     layer_weights = layer.weights
        #     our_model.get_layer("our_head").get_layer(layer_name).set_weights(layer_weights)
        # "all_tracks.h5_head_0"

        # for layer in our_model_old.get_layer("our_hic").layers:
        #     layer_name = layer.name
        #     if "input" in layer_name:
        #         continue
        #     layer_weights = layer.weights
        #     our_model.get_layer("our_hic").get_layer(layer_name).set_weights(layer_weights)

        # our_model.set_weights(joblib.load(model_folder + model_name + "_w"))

        our_model.get_layer("our_resnet").trainable = False
        our_model.get_layer("our_transformer").trainable = False
        # our_model.get_layer("our_hic").trainable = False


        Path(model_folder).mkdir(parents=True, exist_ok=True)
        our_model.save(model_folder + model_name)
        print("Model saved " + model_folder + model_name)
        for head_id in range(len(heads)):
            joblib.dump(our_model.get_layer("our_head").get_weights(),
                        model_folder + model_name + "_head_" + str(head_id), compress=3)
    q.put(None)


def run_epoch(last_proc, fit_epochs, head_id):
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
    # random.shuffle(train_info)
    shuffled_info = random.sample(train_info, len(train_info))
    # shuffled_info = []
    # for info in train_info:
    #     if info[0] == "chr21":
    #         shuffled_info.append(info)
    # shuffled_info.sort(key=lambda x: x[1])

    input_sequences = []
    output_scores = []

    # if k != 0:
    shifts = []
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
    err = 0
    for i, info in enumerate(shuffled_info):
        if len(input_sequences) >= GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH:
            break
        if i % 500 == 0:
            print(i, end=" ")
            gc.collect()
        try:
            shift_bins = random.randint(-int(current_epoch / shift_speed) - initial_shift, int(current_epoch / shift_speed) + initial_shift)
            start = info[1] - (info[1] % bin_size) - half_size + shift_bins * bin_size
            extra = start + input_size - len(one_hot[info[0]])
            if start < 0 or extra > 0:
                shifts.append(None)
                continue
            shifts.append(shift_bins)
            if start < 0:
                ns = one_hot[info[0]][0:start + input_size]
                ns = np.concatenate((np.zeros((-1 * start, num_features)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, num_features))))
            else:
                ns = one_hot[info[0]][start:start + input_size]
            start_bin = int(info[1] / bin_size) - half_num_regions + shift_bins
            scores = []
            for key in heads[head_id]:
                scores.append([info[0], start_bin, start_bin + num_regions])
                # scores.append(gas[key][info[0]][start_bin: start_bin + num_regions])
            input_sequences.append(ns)
            output_scores.append(scores)
        except Exception as e:
            print(e)
            err += 1
    # print("")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")

    start_time = time.time()
    for i, key in enumerate(heads[head_id]):
        if i % 500 == 0:
            end_time = time.time()
            print(f"{i} ({(end_time - start_time):.2f})", end=" ")
            start_time = time.time()
            # gc.collect()
        if key in loaded_tracks.keys():
            # buf = loaded_tracks[key]
            # parsed_track = joblib.load(buf)
            # buf.seek(0)
            parsed_track = loaded_tracks[key]
        else:
            parsed_track = joblib.load(parsed_tracks_folder + key)
        for s in output_scores:
            s[i] = parsed_track[s[i][0]][s[i][1]:s[i][2]].copy()
        # with Pool(4) as p:
        #     map_arr = p.starmap(load_values, zip(output_scores, repeat( [i, parsed_track] )))
    # print("")
    # print(np.asarray(output_scores).shape)
    output_hic = []
    # print(f"Shifts {len(shifts)}")
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Hi-C")
    for hi, key in enumerate(hic_keys):
        print(key, end=" ")
        hdf = joblib.load(parsed_hic_folder + key)
        ni = 0
        for i, info in enumerate(shuffled_info):
            if i >= len(shifts):
                break
            if shifts[i] is None:
                continue
            hd = hdf[info[0]]
            hic_mat = np.zeros((num_hic_bins, num_hic_bins))
            start_hic = int(info[1] - (info[1] % bin_size) - half_size_hic + shifts[i] * bin_size)
            end_hic = start_hic + 2 * half_size_hic
            start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
            end_row = hd['locus1'].searchsorted(end_hic, side='right')
            hd = hd.iloc[start_row:end_row]
            # convert start of the input region to the bin number
            start_hic = int(start_hic / hic_bin_size)
            # subtract start bin from the binned entries in the range [start_row : end_row]
            l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
            l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
            hic_score = hd["score"].values
            # drop contacts with regions outside the [start_row : end_row] range
            lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
            l1 = l1[lix]
            l2 = l2[lix]
            hic_score = hic_score[lix]
            hic_mat[l1, l2] += hic_score
            # hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
            # hic_mat = gaussian_filter(hic_mat, sigma=1)
            # if i == 0:
            #     print(f"original {hic_mat.shape}")
            hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
            # if i == 0:
            #     print(f"triu {hic_mat.shape}")
            # for hs in range(hic_track_size):
            #     hic_slice = hic_mat[hs * num_regions: (hs + 1) * num_regions].copy()
            # if len(hic_slice) != num_regions:
            # hic_mat.resize(num_regions, refcheck=False)
            if hi == 0:
                output_hic.append([])
            output_hic[ni].append(hic_mat)
            ni += 1
        del hd
        del hdf
        gc.collect()
    output_hic = np.asarray(output_hic)
    # print("")
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Problems: " + str(err))
    gc.collect()
    print_memory()
    output_scores = np.asarray(output_scores, dtype=np.float16)
    if np.isnan(output_scores).any() or np.isinf(output_scores).any():
        print("nan in the output")
        exit()
    input_sequences = np.asarray(input_sequences, dtype=np.float16)
    gc.collect()
    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))

    print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
    # gc.collect()
    # print_memory()

    if last_proc is not None:
        print(mp_q.get())
        last_proc.join()
    # joblib.dump([input_sequences, output_scores, output_hic], "pickle/run.gz", compress=3)
    # argss = joblib.load("pickle/run.gz")

    p = mp.Process(target=train_step, args=(input_sequences, output_scores, output_hic, fit_epochs, head_id,))
    p.start()
    return p


def safe_save(thing, place):
    joblib.dump(thing, place + "_temp", compress=3)
    if os.path.exists(place):
        os.remove(place)
    os.rename(place + "_temp", place)


def train_step(input_sequences, output_scores, output_hic, fit_epochs, head_id):
    import tensorflow as tf
    if mixed16:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    for device in physical_devices:
        config1 = tf.config.experimental.set_memory_growth(device, True)

    import model as mo
    if len(output_hic) > 0:
        train_data = mo.wrap_with_hic(input_sequences, [output_scores, output_hic], GLOBAL_BATCH_SIZE)
    else:
        train_data = mo.wrap(input_sequences, output_scores, GLOBAL_BATCH_SIZE)

    zero_fit_1 = np.zeros_like(input_sequences[0])
    zero_fit_2 = np.zeros_like(output_scores[0])
    zero_fit_3 = np.zeros_like(output_hic[0])
    zero_data = mo.wrap_with_hic([zero_fit_1], [[zero_fit_2], [zero_fit_3]], GLOBAL_BATCH_SIZE)

    del input_sequences
    del output_scores
    del output_hic
    # gc.collect()
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Loading the model")
    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + model_name,
                                               custom_objects={'PatchEncoder': mo.PatchEncoder})
        our_model.get_layer("our_head").set_weights(joblib.load(model_folder + model_name + "_head_" + str(head_id)))

        optimizers = [
            tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
        ]

        optimizers_and_layers = [(optimizers[1], our_model.get_layer("our_head")),
                                 (optimizers[0], our_model.get_layer("our_hic")),
                                 (optimizers[0], our_model.get_layer("our_transformer")),
                                 (optimizers[0], our_model.get_layer("our_resnet"))]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

        our_model.get_layer("our_resnet").trainable = False
        our_model.get_layer("our_transformer").trainable = False

        # if current_epoch > 20:
        #     our_model.get_layer("our_transformer").trainable = True
        #
        # if current_epoch > 80:
        #     our_model.get_layer("our_resnet").trainable = True

        loss_weights = {"our_head": 1.0, "our_hic": 0.1}
        our_model.compile(optimizer=optimizer, loss="mse", loss_weights=loss_weights)
        our_model.fit(zero_data)
        if os.path.exists(model_folder + model_name + "_opt_body"):
            print("loading body optimizer")
            optimizers[0].set_weights(joblib.load(model_folder + model_name + "_opt_body"))
            # strategy.run(model_weight_setting(our_model, joblib.load(model_folder + model_name + "_opt_body"), optimizers[0]))
        if os.path.exists(model_folder + model_name + "_opt_head_" + str(head_id)):
            print("loading head optimizer")
            optimizers[1].set_weights(joblib.load(model_folder + model_name + "_opt_head_" + str(head_id)))
            # strategy.run(model_weight_setting(our_model, joblib.load(model_folder + model_name + "_opt_head_" + str(head_id)), optimizers[1]))

    try:
        print(len(our_model.trainable_weights))
        # for var in our_model.trainable_variables:
        #     print(var)
        print(len(our_model.get_layer("our_transformer").trainable_weights))
        print(len(our_model.get_layer("our_resnet").trainable_weights))
    except Exception as e:
        pass

    try:
        our_model.fit(train_data, epochs=fit_epochs)
        # print(datetime.now().strftime('[%H:%M:%S] ') + "Saving " + str(current_epoch) + " model. ")
        our_model.save(model_folder + model_name + "_temp.h5", include_optimizer=False)
        if os.path.exists(model_folder + model_name):
            os.remove(model_folder + model_name)
        os.rename(model_folder + model_name + "_temp.h5", model_folder + model_name)
        safe_save(our_model.get_layer("our_head").get_weights(), model_folder + model_name + "_head_" + str(head_id))
        safe_save(optimizers[0].get_weights(), model_folder + model_name + "_opt_body")
        safe_save(optimizers[1].get_weights(), model_folder + model_name + "_opt_head_" + str(head_id))

        joblib.dump(our_model.get_weights(),
                    model_folder + model_name + "_w", compress=3)
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        mp_q.put(None)
        return None

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " finished. ")
    mp_q.put(None)


def model_weight_setting(model, weight_values, opt):
    import tensorflow as tf
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
    opt.set_weights(weight_values)

def check_perf(mp_q, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    try:
        train_eval_chr = "chr2"
        train_eval_chr_info = []
        for info in train_info:
            if info[0] == train_eval_chr:
                train_eval_chr_info.append(info)
        train_eval_chr_info.sort(key=lambda x: x[1])
        print(f"Training set {len(train_eval_chr_info)}")
        training_spearman = eval_perf(train_eval_chr_info, False, current_epoch, train_eval_chr, head_id)
        print(f"Test set {len(test_info)}")
        test_info.sort(key=lambda x: x[1])
        test_spearman = eval_perf(test_info, True, current_epoch, "chr1", head_id)
        with open(model_name + "_history.csv", "a+") as myfile:
            myfile.write(f"{training_spearman},{test_spearman}")
            myfile.write("\n")
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
    mp_q.put(None)


def eval_perf(eval_infos, should_draw, current_epoch, chr_name, head_id):
    import model as mo
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
    if mixed16:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + model_name,
                                               custom_objects={'PatchEncoder': mo.PatchEncoder})
        our_model.get_layer("our_head").set_weights(joblib.load(model_folder + model_name + "_head_" + str(head_id)))
    predict_batch_size = GLOBAL_BATCH_SIZE
    w_step = 16
    full_preds_steps_num = 10
    eval_track_names = heads[head_id]

    if Path(f"pickle/{chr_name}_seq.gz").is_file():
        test_seq = joblib.load(f"pickle/{chr_name}_seq.gz")
        eval_gt = joblib.load(f"pickle/{chr_name}_eval_gt.gz")
        eval_gt_tss = joblib.load(f"pickle/{chr_name}_eval_gt_tss.gz")
        eval_gt_full = joblib.load(f"pickle/{chr_name}_eval_gt_full.gz")
    else:
        eval_gt = {}
        eval_gt_full = []
        eval_gt_tss = {}
        for i in range(len(eval_infos)):
            eval_gt[eval_infos[i][2]] = {}
            if i < full_preds_steps_num * w_step:
                eval_gt_full.append([])
        for i, key in enumerate(eval_track_names):
            if i % 100 == 0:
                print(i, end=" ")
                gc.collect()
            if key in loaded_tracks.keys():
                # buf = loaded_tracks[key]
                # parsed_track = joblib.load(buf)
                # buf.seek(0)
                parsed_track = loaded_tracks[key]
            else:
                parsed_track = joblib.load(parsed_tracks_folder + key)
            mids = []
            for j, info in enumerate(eval_infos):
                mid = int(info[1] / bin_size)
                # if mid in mids:
                #     continue
                # mids.append(mid)
                # val = parsed_track[info[0]][mid]
                val = parsed_track[info[0]][mid - 1] + parsed_track[info[0]][mid] + parsed_track[info[0]][mid + 1]
                eval_gt_tss.setdefault(key, []).append(val)
                eval_gt[info[2]].setdefault(key, []).append(val)
                if j < full_preds_steps_num * w_step:
                    start_bin = int(info[1] / bin_size) - half_num_regions
                    extra_bin = start_bin + num_regions - len(parsed_track[info[0]])
                    if start_bin < 0:
                        binned_region = parsed_track[info[0]][0: start_bin + num_regions]
                        binned_region = np.concatenate((np.zeros(-1 * start_bin), binned_region))
                    elif extra_bin > 0:
                        binned_region = parsed_track[info[0]][start_bin: len(parsed_track[info[0]])]
                        binned_region = np.concatenate((binned_region, np.zeros(extra_bin)))
                    else:
                        binned_region = parsed_track[info[0]][start_bin:start_bin + num_regions]
                    eval_gt_full[j].append(binned_region)
            if i == 0:
                print(f"Skipped: {len(eval_infos) - len(mids)}")

        for i, gene in enumerate(eval_gt.keys()):
            if i % 10 == 0:
                print(i, end=" ")
            for track in eval_track_names:
                eval_gt[gene][track] = np.mean(eval_gt[gene][track])
        print("")
        eval_gt_full = np.asarray(eval_gt_full)

        test_seq = []
        starts = []
        for info in eval_infos:
            start = int(info[1] - (info[1] % bin_size) - half_size)
            # if start in starts:
            #     continue
            # starts.append(start)
            extra = start + input_size - len(one_hot[info[0]])
            if start < 0:
                ns = one_hot[info[0]][0:start + input_size]
                ns = np.concatenate((np.zeros((-1 * start, num_features)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, num_features))))
            else:
                ns = one_hot[info[0]][start:start + input_size]
            if len(ns) != input_size:
                print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
            test_seq.append(ns)
        print(f"Skipped: {len(eval_infos) - len(starts)}")
        test_seq = np.asarray(test_seq, dtype=bool)
        print(f"Lengths: {len(test_seq)} {len(eval_gt)} {len(eval_gt_full)}")
        joblib.dump(test_seq, f"pickle/{chr_name}_seq.gz", compress="lz4")
        joblib.dump(eval_gt, f"pickle/{chr_name}_eval_gt.gz", compress="lz4")
        joblib.dump(eval_gt_tss, f"pickle/{chr_name}_eval_gt_tss.gz", compress="lz4")
        joblib.dump(eval_gt_full, f"pickle/{chr_name}_eval_gt_full.gz", compress="lz4")

    final_pred = {}
    for i in range(len(eval_infos)):
        final_pred[eval_infos[i][2]] = {}

    for w in range(0, len(test_seq), w_step):
        print(w, end=" ")
        if w != 0 and (w / w_step) % 40 == 0:
            print(" Reloading ")
            gc.collect()
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})

        p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
        # p2 = p1[:, :, mid_bin + correction]
        if len(hic_keys) > 0:
            p2 = p1[0][:, :, mid_bin - 1] + p1[0][:, :, mid_bin] + p1[0][:, :, mid_bin + 1]
            if w == 0:
                predictions = p2
                predictions_full = p1[0]
                predictions_hic = p1[1]
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float32)
                if w / w_step < full_preds_steps_num:
                    predictions_full = np.concatenate((predictions_full, p1[0]), dtype=np.float32)
                    predictions_hic = np.concatenate((predictions_hic, p1[1]), dtype=np.float32)
        else:
            p2 = p1[:, :, mid_bin - 1] + p1[:, :, mid_bin] + p1[:, :, mid_bin + 1]
            # if np.isnan(p2).any() or np.isinf(p2).any():
            #     print("nan predicted")
            #     joblib.dump(test_seq[w:w + w_step], "nan_testseq")
            #     joblib.dump(p2, "nan_testseq")

            if w == 0:
                predictions = p2
                predictions_full = p1
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float32)
                if w / w_step < full_preds_steps_num:
                    predictions_full = np.concatenate((predictions_full, p1), dtype=np.float32)

    final_pred_tss = {}
    for i in range(len(eval_infos)):
        for it, track in enumerate(eval_track_names):
            final_pred[eval_infos[i][2]].setdefault(track, []).append(predictions[i][it])
            final_pred_tss.setdefault(track, []).append(predictions[i][it])

    for i, gene in enumerate(final_pred.keys()):
        if i % 10 == 0:
            print(i, end=" ")
        for track in eval_track_names:
            final_pred[gene][track] = np.mean(final_pred[gene][track])

    corr_p = []
    corr_s = []
    for gene in final_pred.keys():
        a = []
        b = []
        for track in eval_track_names:
            type = track[:track.find(".")]
            if type != "CAGE":
                continue
            # if track not in eval_tracks:
            #     continue
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc) and not math.isnan(pc):
            corr_p.append(pc)
            corr_s.append(sc)

    print("")
    print(f"Maybe this {len(corr_p)} {np.mean(corr_p)} {np.mean(corr_s)}")

    print("Across tracks Genes")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}
    for track in eval_track_names:
        type = track[:track.find(".")]
        a = []
        b = []
        for gene in final_pred.keys():
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        # if "hela" in track.lower():
        #     print(f"{track}\t{sc}")
        track_perf[track] = sc
        if pc is not None and sc is not None:
            corrs_p.setdefault(type, []).append((pc, track))
            corrs_s.setdefault(type, []).append((sc, track))
            all_track_spearman[track] = stats.spearmanr(a, b)[0]

    with open("all_track_spearman.csv", "w+") as myfile:
        for key in all_track_spearman.keys():
            myfile.write(f"{key},{all_track_spearman[key]}")
            myfile.write("\n")

    for track_type in corrs_p.keys():
        with open(f"all_track_spearman_{track_type}.csv", "w+") as myfile:
            for ccc in corrs_s[track_type]:
                myfile.write(f"{ccc[0]},{ccc[1]}")
                myfile.write("\n")
        type_pcc = [i[0] for i in corrs_p[track_type]]
        print(f"{track_type} correlation : {np.mean(type_pcc)}"
              f" {np.mean([i[0] for i in corrs_s[track_type]])} {len(type_pcc)}")
    return_result = np.mean([i[0] for i in corrs_s["CAGE"]])

    with open("result_cage_test.csv", "w+") as myfile:
        for ccc in corrs_p["CAGE"]:
            myfile.write(str(ccc[0]) + "," + str(ccc[1]))
            myfile.write("\n")

    with open("result.txt", "a+") as myfile:
        myfile.write(datetime.now().strftime('[%H:%M:%S] ') + "\n")
        for track_type in corrs_p.keys():
            myfile.write(str(track_type) + "\t")
        for track_type in corrs_p.keys():
            myfile.write(str(np.mean([i[0] for i in corrs_p[track_type]])) + "\t")
        myfile.write("\n")

    # bed_files = {}
    # for i, locus in enumerate(predictions_full):
    #     locus_start = eval_infos[i][1] - half_num_regions * bin_size - (eval_infos[i][1] % bin_size)
    #     for b in range(num_regions):
    #         start = locus_start + b * bin_size
    #         eval_chr = eval_infos[i][0]
    #         for t, track in enumerate(eval_track_names):
    #             type = track[:track.find(".")]
    #             if type != "CAGE":
    #                 continue
    #             bed_files.setdefault(track, []).append(f"{eval_chr}\t{start}\t{start+bin_size}\t{locus[t][b]}\t.\t.\t.")
    #
    # for key in bed_files.keys():
    #     with open("bed_output/" + key + ".bed", 'w+') as f:
    #         f.write('\n'.join(bed_files[key]))

    print("Across tracks TSS")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}
    print(f"Number of TSS: {len(eval_gt_tss[eval_track_names[0]])}")
    for track in eval_track_names:
        type = track[:track.find(".")]
        a = eval_gt_tss[track]
        b = final_pred_tss[track]
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        track_perf[track] = sc
        if pc is not None and sc is not None:
            corrs_p.setdefault(type, []).append((pc, track))
            corrs_s.setdefault(type, []).append((sc, track))
            all_track_spearman[track] = stats.spearmanr(a, b)[0]

    with open("all_track_spearman_tss.csv", "w+") as myfile:
        for key in all_track_spearman.keys():
            myfile.write(f"{key},{all_track_spearman[key]}")
            myfile.write("\n")

    for track_type in corrs_p.keys():
        with open(f"all_track_spearman_{track_type}_tss.csv", "w+") as myfile:
            for ccc in corrs_s[track_type]:
                myfile.write(f"{ccc[0]},{ccc[1]}")
                myfile.write("\n")
        type_pcc = [i[0] for i in corrs_p[track_type]]
        print(f"{track_type} correlation : {np.mean(type_pcc)}"
              f" {np.mean([i[0] for i in corrs_s[track_type]])} {len(type_pcc)}")
    return_result = np.mean([i[0] for i in corrs_s["CAGE"]])

    if should_draw:
        hic_output = []
        for hi, key in enumerate(hic_keys):
            hdf = joblib.load(parsed_hic_folder + key)
            ni = 0
            for i, info in enumerate(eval_infos):
                hd = hdf[info[0]]
                hic_mat = np.zeros((num_hic_bins, num_hic_bins))
                start_hic = int(info[1] - (info[1] % bin_size) - half_size_hic)
                end_hic = start_hic + 2 * half_size_hic
                start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
                end_row = hd['locus1'].searchsorted(end_hic, side='right')
                hd = hd.iloc[start_row:end_row]
                # convert start of the input region to the bin number
                start_hic = int(start_hic / hic_bin_size)
                # subtract start bin from the binned entries in the range [start_row : end_row]
                l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
                l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
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

        draw_arguments = [eval_track_names, track_perf, predictions_full, eval_gt_full,
                        test_seq, bin_size, num_regions, eval_infos, hic_keys,
                        hic_track_size, predictions_hic, hic_output,
                        num_hic_bins, f"{figures_folder}/tracks/epoch_{current_epoch}"]
        joblib.dump(draw_arguments, "draw", compress=3)

        viz.draw_tracks(eval_track_names, track_perf, predictions_full, eval_gt_full,
                        test_seq, bin_size, num_regions, eval_infos, hic_keys,
                        hic_track_size, predictions_hic, hic_output,
                        num_hic_bins, f"{figures_folder}/tracks/epoch_{current_epoch}")

        viz.draw_regplots(eval_track_names, track_perf, final_pred, eval_gt,
                          f"{figures_folder}/plots/epoch_{current_epoch}")

        viz.draw_attribution()

    return return_result


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def change_seq(x):
    return cm.rev_comp(x)


def load_values(s, chosen_track):
    i = chosen_track[0]
    parsed_track = chosen_track[1]
    s[i] = parsed_track[s[i][0]][s[i][1]:s[i][2]].copy()
    return 1


if __name__ == '__main__':
    # import model as mo
    # our_model = mo.hic_model(input_size, num_features, num_regions, out_stack_num, 2, hic_size)
    script_folder = pathlib.Path(__file__).parent.resolve()
    folders = open(str(script_folder) + "/data_dirs").read().strip().split("\n")
    os.chdir(folders[0])
    parsed_tracks_folder = folders[1]
    parsed_hic_folder = folders[2]
    model_folder = folders[3]

    # draw = joblib.load("draw")
    # viz.draw_tracks(*draw)

    # os.chdir("/home/acd13586qv/variants")
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "plots").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)
    ga, one_hot, train_info, test_info, tss_loc = parser.get_sequences(bin_size, chromosomes)
    if Path("pickle/track_names.gz").is_file():
        track_names = joblib.load("pickle/track_names.gz")
    else:
        track_names = parser.parse_tracks(ga, bin_size, tss_loc, chromosomes, tracks_folder)
    # track_names = track_names[:100]
    print("Number of tracks: " + str(len(track_names)))

    if Path("pickle/heads.gz").is_file():
        heads = joblib.load("pickle/heads.gz")
    else:
        heads = []
        random.shuffle(track_names)
        heads.append(track_names[:4000])
        heads.append(track_names[4000:8000])
        heads.append(track_names[8000:])
        random.shuffle(track_names)
        heads[-1].extend(track_names[:4000 - len(heads[-1])])
        joblib.dump(heads, "pickle/heads.gz", compress=3)

    for head in heads:
        print(len(head))
    # hic_keys = parser.parse_hic()
    hic_keys = ["hic_ADAC418_10kb_interactions.txt.bz2"]  # , "hic_A549_10kb_interactions.txt.bz2"
    # "hic_HepG2_10kb_interactions.txt.bz2", "hic_THP1_10kb_interactions.txt.bz2"]
    hic_num = len(hic_keys)
    # hic_keys = []

    mp_q = mp.Queue()
    # create_model(mp_q)
    # exit()

    if not Path(model_folder + model_name).is_file():
        p = mp.Process(target=create_model, args=(mp_q,))
        p.start()
        print(mp_q.get())
        p.join()
        time.sleep(1)
    else:
        print("Model exists")

    loaded_tracks = {}
    # for i, key in enumerate(track_names):
    #     if i % 100 == 0:
    #         print(i, end=" ")
    #         # gc.collect()
    #     # with open(parsed_tracks_folder + key, 'rb') as fh:
    #     #     loaded_tracks[key] = io.BytesIO(fh.read())
    #     loaded_tracks[key] = joblib.load(parsed_tracks_folder + key)
    #     if i > 1000:
    #         break

    # mp.set_start_method('spawn', force=True)
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    # p = mp.Process(target=recompile, args=(q,))
    # p.start()
    # print(q.get())
    # p.join()
    # time.sleep(1)
    print("Training starting")
    start_epoch = 0
    fit_epochs = 4
    for current_epoch in range(start_epoch, num_epochs, 1):
        head_id = current_epoch % len(heads)
        # if current_epoch < 10:
        #     fit_epochs = 1
        # elif current_epoch < 100:
        #     fit_epochs = 2
        # else:
        #     fit_epochs = 4

        # if current_epoch == 60:
        #     if last_proc is not None:
        #         print(mp_q.get())
        #         last_proc.join()
        #     print("Recompiling")
        #     p = mp.Process(target=recompile, args=(mp_q, 0.0001,1,))
        #     p.start()
        #     print(mp_q.get())
        #     p.join()
        #     time.sleep(1)
        #     last_proc = None

        # check_perf()
        # exit()
        last_proc = run_epoch(last_proc, fit_epochs, head_id)
        if current_epoch % 10 == 0: # and current_epoch != 0:
            print(mp_q.get())
            last_proc.join()
            last_proc = None
            p = mp.Process(target=check_perf, args=(mp_q, head_id,))
            p.start()
            print(mp_q.get())
            p.join()
