import io
import os

import joblib

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import logging

# logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
import pickle
import model as mo
matplotlib.use("agg")
from scipy.ndimage.filters import gaussian_filter
# from sam import SAM
from itertools import repeat

# tf.compat.v1.disable_eager_execution()
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# for device in physical_devices:
#     config1 = tf.config.experimental.set_memory_growth(device, True)

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
input_size = 204001
half_size = int(input_size / 2)
bin_size = 200
hic_bin_size = 10000
num_hic_bins = int(input_size / hic_bin_size)
num_regions = 1001  # int(input_size / bin_size)
half_num_regions = int(num_regions / 2)
mid_bin = math.floor(num_regions / 2)
BATCH_SIZE = 1
STEPS_PER_EPOCH = 100
chromosomes = ["chrX"]  # "chrY"
for i in range(1, 23):
    chromosomes.append("chr" + str(i))
num_epochs = 1000
hic_track_size = 1
out_stack_num = 9137 + 1 * hic_track_size
num_features = 5


def recompile(q):
    import tensorflow as tf
    import model as mo

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                               custom_objects={'SAMModel': mo.SAMModel,
                                                               'PatchEncoder': mo.PatchEncoder})
        print(datetime.now().strftime('[%H:%M:%S] ') + "Compiling model")
        lr = 0.0005
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            our_model.compile(loss="mse", optimizer=optimizer)

        our_model.save(model_folder + "/" + model_name)
        print("Model saved " + model_folder + "/" + model_name)
    q.put(None)


def create_model(q):
    import tensorflow as tf

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    import model as mo
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = mo.simple_model(input_size, num_regions, out_stack_num)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Compiling model")
        lr = 0.0001
        with strategy.scope():
            # base_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            # base_optimizer = LossScaleOptimizer(base_optimizer, initial_scale=2 ** 2)
            # optimizer = SAM(base_optimizer)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            our_model.compile(loss="mse", optimizer=optimizer)

        Path(model_folder).mkdir(parents=True, exist_ok=True)
        our_model.save(model_folder + "/" + model_name)
        print("Model saved " + model_folder + "/" + model_name)
    q.put(None)


def run_epoch(q, k, train_info, test_info, one_hot, track_names, loaded_tracks, hic_keys):
    import tensorflow as tf
    import model as mo

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # for device in physical_devices:
    #     config1 = tf.config.experimental.set_memory_growth(device, True)

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k))
    random.shuffle(train_info)

    input_sequences = []
    output_scores = []
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading the model")

    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                               custom_objects={'SAMModel': mo.SAMModel,
                                                               'PatchEncoder': mo.PatchEncoder})

    # if k != 0:
    shifts = []
    print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
    err = 0
    for i, info in enumerate(train_info):
        if len(input_sequences) >= GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH:
            break
        if i % 500 == 0:
            print(i, end=" ")
            gc.collect()
        try:
            shift_bins = random.randint(-int(k / 5), int(k / 5))
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
            start_bin = int(info[1] / bin_size) - half_num_regions
            scores = []
            for key in track_names:
                scores.append([info[0], start_bin, start_bin + num_regions])
                # scores.append(gas[key][info[0]][start_bin: start_bin + num_regions])
            input_sequences.append(ns)
            output_scores.append(scores)
        except Exception as e:
            print(e)
            err += 1
    print("")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")

    for i, key in enumerate(track_names):
        if i % 100 == 0:
            print(i, end=" ")
            # gc.collect()
        if key in loaded_tracks.keys():
            buf = loaded_tracks[key]
            parsed_track = joblib.load(buf)
            buf.seek(0)
        else:
            parsed_track = joblib.load(parsed_tracks_folder + key)
        for s in output_scores:
            s[i] = parsed_track[s[i][0]][s[i][1]:s[i][2]].copy()
        # with Pool(4) as p:
        #     map_arr = p.starmap(load_values, zip(output_scores, repeat( [i, parsed_track] )))
    print("")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Hi-C")
    for key in hic_keys:
        print(key, end=" ")
        hdf = joblib.load(parsed_hic_folder + key)
        for i, info in enumerate(train_info):
            if i >= GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH:
                break
            if shifts[i] is None:
                continue
            hd = hdf[info[0]]
            hic_mat = np.zeros((num_hic_bins, num_hic_bins))
            start_hic = int((info[1] - half_size + shifts[i] * bin_size))
            end_hic = start_hic + input_size
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
            hic_mat = gaussian_filter(hic_mat, sigma=1)
            if i == 0:
                print(f"original {len(hic_mat.flatten())}")
            hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
            if i == 0:
                print(f"triu {len(hic_mat.flatten())}")
            for hs in range(hic_track_size):
                hic_slice = hic_mat[hs * num_regions: (hs + 1) * num_regions].copy()
                if len(hic_slice) != num_regions:
                    hic_slice.resize(num_regions, refcheck=False)
                output_scores[i].append(hic_slice)
        del hd
        del hdf
        gc.collect()

    print(datetime.now().strftime('[%H:%M:%S] ') + "Problems: " + str(err))
    gc.collect()
    print_memory()
    output_scores = np.asarray(output_scores, dtype=np.float16)
    input_sequences = np.asarray(input_sequences, dtype=np.float16)
    gc.collect()
    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))

    print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
    gc.collect()
    print_memory()

    try:
        fit_epochs = 1
        train_data = mo.wrap(input_sequences, output_scores, GLOBAL_BATCH_SIZE)
        gc.collect()
        our_model.fit(train_data, epochs=fit_epochs)
        our_model.save(model_folder + "/" + model_name + "_temp.h5")
        os.remove(model_folder + "/" + model_name)
        os.rename(model_folder + "/" + model_name + "_temp.h5", model_folder + "/" + model_name)
        our_model.save(model_folder + "/" + model_name + "_no.h5", include_optimizer=False)
        del train_data
        gc.collect()
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        q.put(None)
        return None
    if k % 10 == 0:  # and k != 0
        print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
        try:
            train_eval_chr = "chr2"
            train_eval_chr_info = []
            for info in train_info:
                if info[0] == train_eval_chr:
                    train_eval_chr_info.append(info)
            print(f"Training set {len(train_eval_chr_info)}")
            training_spearman = eval_perf(strategy, GLOBAL_BATCH_SIZE, train_eval_chr_info, loaded_tracks, False, k, train_eval_chr)
            print(f"Test set {len(test_info)}")
            test_spearman = eval_perf(strategy, GLOBAL_BATCH_SIZE, test_info, loaded_tracks, True, k, "chr1")
            with open(model_name + "_history.csv", "a+") as myfile:
                myfile.write(f"{training_spearman},{test_spearman}")
                myfile.write("\n")
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k) + " finished. ")
    q.put(None)


def eval_perf(strategy, GLOBAL_BATCH_SIZE, eval_infos, loaded_tracks, should_draw, current_epoch, chr_name):
    import model as mo
    import tensorflow as tf
    from tensorflow.python.keras import backend as K

    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + "/" + model_name + "_no.h5",
                                               custom_objects={'SAMModel': mo.SAMModel,
                                                               'PatchEncoder': mo.PatchEncoder})
    predict_batch_size = GLOBAL_BATCH_SIZE
    w_step = 64
    full_preds_steps_num = 2

    if Path(f"pickle/{chr_name}_seq.gz").is_file():
        test_seq = joblib.load(f"pickle/{chr_name}_seq.gz")
        eval_gt = joblib.load(f"pickle/{chr_name}_eval_gt.gz")
        eval_gt_full = joblib.load(f"pickle/{chr_name}_eval_gt_full.gz")
    else:
        eval_gt = {}
        eval_gt_full = []
        for i in range(len(eval_infos)):
            eval_gt[eval_infos[i][2]] = {}
            if i < full_preds_steps_num * w_step:
                eval_gt_full.append([])
        for i, key in enumerate(track_names):
            if i % 100 == 0:
                print(i, end=" ")
                gc.collect()
            if key in loaded_tracks.keys():
                buf = loaded_tracks[key]
                parsed_track = joblib.load(buf)
                buf.seek(0)
            else:
                parsed_track = joblib.load(parsed_tracks_folder + key)

            for j, info in enumerate(eval_infos):
                mid = int(info[1] / bin_size)
                # val = parsed_track[info[0]][mid]
                val = parsed_track[info[0]][mid - 1] + parsed_track[info[0]][mid] + parsed_track[info[0]][mid + 1]
                eval_gt[info[2]].setdefault(key, []).append(val)
                if j >= w_step and len(eval_gt_full) < full_preds_steps_num * w_step:
                    start_bin = int(info[1] / bin_size) - half_num_regions
                    eval_gt_full[j - w_step].append(parsed_track[info[0]][start_bin:start_bin + num_regions])

        for i, gene in enumerate(eval_gt.keys()):
            if i % 10 == 0:
                print(i, end=" ")
            for track in track_names:
                eval_gt[gene][track] = np.mean(eval_gt[gene][track])
        print("")
        eval_gt_full = np.asarray(eval_gt_full)

        test_seq = []
        for info in eval_infos:
            start = int(info[1] - (info[1] % bin_size) - half_size)
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

        test_seq = np.asarray(test_seq, dtype=np.float16)
        print(f"Lengths: {len(test_seq)} {len(eval_gt)} {len(eval_gt_full)}")
        joblib.dump(test_seq, f"pickle/{chr_name}_seq.gz", compress="lz4")
        joblib.dump(eval_gt, f"pickle/{chr_name}_eval_gt.gz", compress="lz4")
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
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name + "_no.h5",
                                                       custom_objects={'SAMModel': mo.SAMModel,
                                                                       'PatchEncoder': mo.PatchEncoder})

        p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
        # p2 = p1[:, :, mid_bin + correction]
        p2 = p1[:, :, mid_bin - 1] + p1[:, :, mid_bin] + p1[:, :, mid_bin + 1]

        if w == 0:
            predictions = p2
        else:
            predictions = np.concatenate((predictions, p2), dtype=np.float16)
            if w / w_step == 1:
                predictions_full = p1
            elif w / w_step < full_preds_steps_num:
                predictions_full = np.concatenate((predictions_full, p1), dtype=np.float16)

    for i in range(len(eval_infos)):
        for it, ct in enumerate(track_names):
            final_pred[eval_infos[i][2]].setdefault(ct, []).append(predictions[i][it])

    for i, gene in enumerate(final_pred.keys()):
        if i % 10 == 0:
            print(i, end=" ")
        for track in track_names:
            final_pred[gene][track] = np.mean(final_pred[gene][track])

    corr_p = []
    corr_s = []
    for gene in final_pred.keys():
        a = []
        b = []
        for track in track_names:
            type = track[:track.find(".")]
            if type != "CAGE":
                continue
            # if track not in eval_tracks:
            #     continue
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc) and not math.isnan(pc):
            corr_p.append(pc)
            corr_s.append(sc)

    print("")
    print(f"Maybe this {len(corr_p)} {np.mean(corr_p)} {np.mean(corr_s)}")

    print("Across tracks")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    for track in track_names:
        type = track[:track.find(".")]
        a = []
        b = []
        for gene in final_pred.keys():
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
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
        print(
            f"{track_type} correlation : {np.mean([i[0] for i in corrs_p[track_type]])} {np.mean([i[0] for i in corrs_s[track_type]])}")
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

    if should_draw:
        print("Drawing tracks")
        pic_count = 0
        for it, ct in enumerate(track_names):
            type = ct[:ct.find(".")]
            if type != "CAGE":
                continue
            for i in range(len(predictions_full)):
                if np.sum(eval_gt_full[i][it]) == 0:
                    continue
                fig, axs = plt.subplots(2, 1, figsize=(12, 8))
                vector1 = predictions_full[i][it]
                vector2 = eval_gt_full[i][it]
                x = range(num_regions)
                d1 = {'bin': x, 'expression': vector1}
                df1 = pd.DataFrame(d1)
                d2 = {'bin': x, 'expression': vector2}
                df2 = pd.DataFrame(d2)
                sns.lineplot(data=df1, x='bin', y='expression', ax=axs[0])
                axs[0].set_title("Prediction")
                sns.lineplot(data=df2, x='bin', y='expression', ax=axs[1])
                axs[1].set_title("Ground truth")
                fig.tight_layout()
                plt.savefig(f"{figures_folder}/tracks/epoch_{current_epoch}_{eval_infos[i][2]}_{ct}.png")
                plt.close(fig)
                pic_count += 1
                if i > 20:
                    break
            if pic_count > 100:
                break

        pic_count = 0
        print("Drawing gene regplot")
        for it, track in enumerate(track_names):
            type = track[:track.find(".")]
            if type != "CAGE":
                continue
            a = []
            b = []
            for gene in final_pred.keys():
                a.append(final_pred[gene][track])
                b.append(eval_gt[gene][track])

            fig, ax = plt.subplots(figsize=(6, 6))
            r, p = stats.spearmanr(a, b)

            sns.regplot(x=a, y=b,
                        ci=None, label="r = {0:.2f}; p = {1:.2e}".format(r, p)).legend(loc="best")

            ax.set(xlabel='Predicted', ylabel='Ground truth')
            plt.title("Gene expression prediction")
            fig.tight_layout()
            plt.savefig(f"{figures_folder}/plots/epoch_{current_epoch}_{track}.svg")
            plt.close(fig)
            pic_count += 1
            if pic_count > 100:
                break

        # attribution
        # for c, cell in enumerate(cells):
        #     for i in range(1200, 1210, 1):
        #         baseline = tf.zeros(shape=(input_size, num_features))
        #         image = test_input_sequences[i].astype('float32')
        #         ig_attributions = attribution.integrated_gradients(our_model, baseline=baseline,
        #                                                            image=image,
        #                                                            target_class_idx=[mid_bin, c],
        #                                                            m_steps=40)
        #
        #         attribution_mask = tf.squeeze(ig_attributions).numpy()
        #         attribution_mask = (attribution_mask - np.min(attribution_mask)) / (
        #                     np.max(attribution_mask) - np.min(attribution_mask))
        #         attribution_mask = np.mean(attribution_mask, axis=-1, keepdims=True)
        #         attribution_mask[int(input_size / 2) - 2000 : int(input_size / 2) + 2000, :] = np.nan
        #         attribution_mask = skimage.measure.block_reduce(attribution_mask, (100, 1), np.mean)
        #         attribution_mask = np.transpose(attribution_mask)
        #
        #         fig, ax = plt.subplots(figsize=(60, 6))
        #         sns.heatmap(attribution_mask, linewidth=0.0, ax=ax)
        #         plt.tight_layout()
        #         plt.savefig(figures_folder + "/attribution/track_" + str(i + 1) + "_" + str(cell) + "_" + test_info[i] + ".jpg")
        #         plt.close(fig)
    else:
        hic_output = []
        del final_pred
        del eval_gt
        del eval_gt_full
        gc.collect()
        print("\nHi-C")
        for key in hic_keys:
            print(key, end=" ")
            hdf = joblib.load(parsed_hic_folder + key)
            ni = 0
            hic_output.append([])
            for i, info in enumerate(eval_infos):
                hd = hdf[info[0]]
                hic_mat = np.zeros((num_hic_bins, num_hic_bins))
                start_hic = int((info[1] - half_size))
                end_hic = start_hic + input_size
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
                hic_mat = gaussian_filter(hic_mat, sigma=1)
                # print(f"original {len(hic_mat.flatten())}")
                hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
                # print(f"triu {len(hic_mat.flatten())}")
                for hs in range(hic_track_size):
                    hic_slice = hic_mat[hs * num_regions: (hs + 1) * num_regions].copy()
                    if len(hic_slice) != num_regions:
                        hic_slice.resize(num_regions, refcheck=False)
                    hic_output[ni].append(hic_slice)
                ni += 1
                hic_output.append([])
            del hd
            del hdf
            gc.collect()
        print("Drawing contact maps")
        for h in range(len(hic_keys)):
            pic_count = 0
            it = h * hic_track_size
            it2 = len(track_names) + h * hic_track_size
            for i in range(len(predictions_full)):
                mat_gt = recover_shape(hic_output[i][it:it + hic_track_size], num_hic_bins)
                mat_pred = recover_shape(predictions_full[i][it2:it2 + hic_track_size], num_hic_bins)
                fig, axs = plt.subplots(2, 1, figsize=(6, 8))
                sns.heatmap(mat_pred, linewidth=0.0, ax=axs[0])
                axs[0].set_title("Prediction")
                sns.heatmap(mat_gt, linewidth=0.0, ax=axs[1])
                axs[1].set_title("Ground truth")
                plt.tight_layout()
                plt.savefig(figures_folder + "/hic/train_track_" + str(i + 1) + "_" + str(hic_keys[h]) + ".png")
                plt.close(fig)
                pic_count += 1
                if pic_count > 5:
                    break
    return return_result


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def recover_shape(v, size_X):
    v = np.asarray(v).flatten()
    end = int((size_X * size_X - size_X) / 2)
    v = v[:end]
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=1)] = v
    X = X + X.T
    return X


def change_seq(x):
    return cm.rev_comp(x)


def load_values(s, chosen_track):
    i = chosen_track[0]
    parsed_track = chosen_track[1]
    s[i] = parsed_track[s[i][0]][s[i][1]:s[i][2]].copy()
    return 1


model_folder = "models"
model_name = "1mb.h5"
figures_folder = "figures_1"
tracks_folder = "/home/user/data/tracks/"
temp_folder = "/home/user/data/temp/"

# parsed_tracks_folder = "/home/user/data/parsed_tracks/"
# parsed_hic_folder = "/home/user/data/parsed_hic/"

parsed_tracks_folder = "parsed_tracks/"
parsed_hic_folder = "parsed_hic/"

# temp_folder = "temp/"

if __name__ == '__main__':
    # our_model = mo.simple_model(input_size, num_regions, out_stack_num)
    # os.chdir(open("data_dir").read().strip())
    os.chdir("/home/acd13586qv/variants")
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

    print("Number of tracks: " + str(len(track_names)))

    # hic_keys = parser.parse_hic()
    hic_keys = ["hic_ADAC418_10kb_interactions.txt.bz2"]
    # hic_keys = []

    loaded_tracks = {}
    for i, key in enumerate(track_names):
        if i % 100 == 0:
            print(i, end=" ")
            # gc.collect()
        with open(parsed_tracks_folder + key, 'rb') as fh:
            loaded_tracks[key] = io.BytesIO(fh.read())

    # mp.set_start_method('spawn', force=True)
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    q = mp.Queue()
    if not Path(model_folder + "/" + model_name).is_file():
        p = mp.Process(target=create_model, args=(q,))
        p.start()
        print(q.get())
        p.join()
        time.sleep(1)
    else:
        print("Model exists")
    # p = mp.Process(target=recompile, args=(q,))
    # p.start()
    # print(q.get())
    # p.join()
    # time.sleep(1)
    print("Training starting")
    for k in range(num_epochs):
        # if k == 10:
        #     fit_epochs = 1
        #     lr = 0.0004
        #     p = mp.Process(target=recompile, args=(q, lr,))
        #     p.start()
        #     print(q.get())
        #     p.join()
        #     time.sleep(1)
        p = mp.Process(target=run_epoch, args=(q, k, train_info, test_info, one_hot, track_names, loaded_tracks, hic_keys,))
        p.start()
        print(q.get())
        p.join()
        time.sleep(1)
