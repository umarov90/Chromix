import math
import os
import logging
import joblib
import gc
import random
import time
import numpy as np
import common as cm
from pathlib import Path
import matplotlib
import psutil
import sys
import parse_data as parser
from datetime import datetime
import traceback
import multiprocessing as mp
import evaluation
from main_params import MainParams
import tensorflow_addons as tfa
from scipy.ndimage import gaussian_filter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
matplotlib.use("agg")


def run_epoch(last_proc, fit_epochs, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " " + p.species[head_id])
    training_regions = joblib.load(f"pickle/{p.species[head_id]}_regions.gz")
    one_hot = joblib.load(f"pickle/{p.species[head_id]}_one_hot.gz")
    shuffled_info = random.sample(training_regions, len(training_regions))
    input_sequences = []
    output_scores_info = []
    shifts = []
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
    # pick training regions
    for i, info in enumerate(shuffled_info):
        if len(input_sequences) >= p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH:
            break
        if i % 500 == 0:
            print(i, end=" ")
            gc.collect()
        shift_bins = random.randint(-int(current_epoch / p.shift_speed) - p.initial_shift,
                                    int(current_epoch / p.shift_speed) + p.initial_shift)
        start = info[1] - (info[1] % p.bin_size) - p.half_size + shift_bins * p.bin_size
        extra = start + p.input_size - len(one_hot[info[0]])
        if start < 0 or extra > 0:
            shifts.append(None)
            continue
        ns = one_hot[info[0]][start:start + p.input_size]
        if np.any(ns[:, -1]):
            shifts.append(None)
            continue

        picked_training_regions.setdefault(p.species[head_id], []).append(
            f"{info[0]}\t{start}\t{start + p.input_size}\ttrain")
        shifts.append(shift_bins)
        # if start < 0:
        #     ns = one_hots[head_id][info[0]][0:start + p.input_size]
        #     ns = np.concatenate((np.zeros((-1 * start, p.num_features)), ns))
        # elif extra > 0:
        #     ns = one_hots[head_id][info[0]][start: len(one_hots[head_id][info[0]])]
        #     ns = np.concatenate((ns, np.zeros((extra, p.num_features))))
        # else:

        start_bin = int(info[1] / p.bin_size) - p.half_num_regions + shift_bins
        scores = []
        for key in heads[head_id]:
            scores.append([info[0], start_bin, start_bin + p.num_bins])
            # scores.append(gas[key][info[0]][start_bin: start_bin + num_regions])
        input_sequences.append(ns)
        output_scores_info.append(scores)
    # print("")
    # return 1
    output_scores_info = np.asarray(output_scores_info)
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")

    ps = []
    start = 0
    nproc = min(mp.cpu_count(), len(heads[head_id]))
    step_size = len(heads[head_id]) // nproc
    end = len(heads[head_id])
    for t in range(start, end, step_size):
        t_end = min(t + step_size, end)
        load_proc = mp.Process(target=load_data,
                               args=(mp_q, p, heads[head_id][t:t_end], output_scores_info[:, t:t_end], t, t_end,))
        load_proc.start()
        ps.append(load_proc)

    for load_proc in ps:
        load_proc.join()
    print(mp_q.get())

    output_scores = []
    for t in range(start, end, step_size):
        output_scores.append(joblib.load(f"temp/data{t}"))

    gc.collect()
    output_scores = np.concatenate(output_scores, axis=1, dtype=np.float32)
    gc.collect()
    print("")
    half = len(input_sequences) // 2
    output_hic = []
    # print(f"Shifts {len(shifts)}")
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Hi-C")
    if head_id == 0:
        for hi, key in enumerate(hic_keys):
            print(key, end=" ")
            hdf = joblib.load(p.parsed_hic_folder + key)
            ni = 0
            for i, info in enumerate(shuffled_info):
                if i >= len(shifts):
                    break
                if shifts[i] is None:
                    continue
                hd = hdf[info[0]]
                hic_mat = np.zeros((p.num_hic_bins, p.num_hic_bins))
                start_hic = int(info[1] - (info[1] % p.bin_size) - p.half_size_hic + shifts[i] * p.bin_size)
                end_hic = start_hic + 2 * p.half_size_hic
                start_row = hd['start1'].searchsorted(start_hic - p.hic_bin_size, side='left')
                end_row = hd['start1'].searchsorted(end_hic, side='right')
                hd = hd.iloc[start_row:end_row]
                # convert start of the input region to the bin number
                start_hic = int(start_hic / p.hic_bin_size)
                # subtract start bin from the binned entries in the range [start_row : end_row]
                l1 = (np.floor(hd["start1"].values / p.hic_bin_size) - start_hic).astype(int)
                l2 = (np.floor(hd["start2"].values / p.hic_bin_size) - start_hic).astype(int)
                hic_score = hd["score"].values
                # drop contacts with regions outside the [start_row : end_row] range
                lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
                l1 = l1[lix]
                l2 = l2[lix]
                hic_score = hic_score[lix]
                hic_mat[l1, l2] += hic_score
                hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
                hic_mat = gaussian_filter(hic_mat, sigma=0.5)
                if ni < half:
                    hic_mat = np.rot90(hic_mat, k=2)
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
    if np.isnan(output_scores).any() or np.isinf(output_scores).any():
        print("nan in the output")
        exit()
    input_sequences = np.asarray(input_sequences, dtype=bool)

    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr

    for i in range(half):
        output_scores[i] = np.flip(output_scores[i], axis=1)

    # Cut off the test TSS layer
    input_sequences = input_sequences[:, :, :-1]
    print(input_sequences.shape)

    gc.collect()
    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))

    print(datetime.now().strftime('[%H:%M:%S] ') + "Training")

    if last_proc is not None:
        print("Waiting")
        print(mp_q.get())
        last_proc.join()
        print("Finished waiting")
    # joblib.dump([input_sequences, output_scores, output_hic], "pickle/run.gz", compress=3)
    # argss = joblib.load("pickle/run.gz")
    # p = mp.Process(target=train_step, args=(argss[0][:400], argss[1][:400], argss[2][:400], fit_epochs, head_id,))
    proc = mp.Process(target=train_step, args=(
        heads[head_id], p.species[head_id], input_sequences, output_scores, output_hic, fit_epochs, hic_num, mp_q,))
    proc.start()
    return proc


def safe_save(thing, place):
    joblib.dump(thing, place + "_temp", compress="lz4")
    if os.path.exists(place):
        os.remove(place)
    os.rename(place + "_temp", place)


def train_step(head, head_name, input_sequences, output_scores, output_hic, fit_epochs, hic_num, mp_q):
    hic_step = True
    if head_name != "hg38" or hic_num == 0:
        hic_step = False
    try:
        import tensorflow as tf
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        # for device in physical_devices:
        #     tf.config.experimental.set_memory_growth(device, True)

        import model as mo
        if hic_step:
            train_data = mo.wrap_with_hic(input_sequences, [output_scores, output_hic], p.GLOBAL_BATCH_SIZE)
            zero_fit_1 = np.zeros_like(input_sequences[0])
            zero_fit_2 = np.zeros_like(output_scores[0])
            zero_fit_3 = np.zeros_like(output_hic[0])
            zero_data = mo.wrap_with_hic([zero_fit_1], [[zero_fit_2], [zero_fit_3]], p.GLOBAL_BATCH_SIZE)
        else:
            train_data = mo.wrap(input_sequences, output_scores, p.GLOBAL_BATCH_SIZE)
            zero_fit_1 = np.zeros_like(input_sequences[0])
            zero_fit_2 = np.zeros_like(output_scores[0])
            zero_data = mo.wrap([zero_fit_1], [zero_fit_2], p.GLOBAL_BATCH_SIZE)

        del input_sequences
        del output_scores
        del output_hic
        gc.collect()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        # print(datetime.now().strftime('[%H:%M:%S] ') + "Loading the model")
        with strategy.scope():
            if hic_step:
                our_model = mo.hic_model(p.input_size, p.num_features, p.num_bins, len(head), hic_num, p.hic_size,
                                         p.bin_size)
            else:
                our_model = mo.small_model(p.input_size, p.num_features, p.num_bins, len(head), p.bin_size)
            print(f"=== Training with head {head_name} ===")
            hic_lr = 0.0001
            head_lr = 0.0001
            resnet_lr = 0.00001
            resnet_wd = 0.0000001
            # cap_e = min(current_epoch, 40)
            # transformer_lr = 0.0000005 + cap_e * 0.0000025
            # tfa.optimizers.AdamW
            # optimizers = [
            #     tf.keras.optimizers.Adam(learning_rate=resnet_lr),
            #     tfa.optimizers.AdamW(learning_rate=transformer_lr, weight_decay=transformer_wd),
            #     tf.keras.optimizers.Adam(learning_rate=head_lr)
            # ]
            optimizers = {"our_resnet": tfa.optimizers.AdamW(learning_rate=resnet_lr, weight_decay=resnet_wd),
                          "our_head": tf.keras.optimizers.Adam(learning_rate=head_lr)}

            optimizers_and_layers = [(optimizers["our_resnet"], our_model.get_layer("our_resnet")),
                                     (optimizers["our_head"], our_model.get_layer("our_head"))
                                     ]
            if hic_step:
                optimizers["our_hic"] = tf.keras.optimizers.Adam(learning_rate=hic_lr)
                optimizers_and_layers.append((optimizers["our_hic"], our_model.get_layer("our_hic")))

            optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

            our_model.get_layer("our_resnet").trainable = True

            if hic_step:
                loss_weights = {"our_head": 1.0, "our_hic": 0.1}
                losses = {
                    "our_head": "mse",
                    "our_hic": "mse",
                }
                our_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
            else:
                our_model.compile(optimizer=optimizer, loss="mse")

            our_model.fit(zero_data, steps_per_epoch=1, epochs=1)

            if os.path.exists(p.model_path + "_res"):
                our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            if os.path.exists(p.model_path + "_head_" + head_name):
                our_model.get_layer("our_head").set_weights(joblib.load(p.model_path + "_head_" + head_name))
            if hic_step and os.path.exists(p.model_path + "_hic"):
                our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

            if os.path.exists(p.model_path + "_opt_resnet"):
                print("loading resnet optimizer")
                optimizers["our_resnet"].set_weights(joblib.load(p.model_path + "_opt_resnet"))

            if os.path.exists(p.model_path + "_opt_head_" + head_name):
                print("loading head optimizer")
                optimizers["our_head"].set_weights(joblib.load(p.model_path + "_opt_head_" + head_name))

            if hic_step and os.path.exists(p.model_path + "_opt_hic"):
                print("loading hic optimizer")
                optimizers["our_hic"].set_weights(joblib.load(p.model_path + "_opt_hic"))

            optimizers["our_resnet"].learning_rate = resnet_lr
            optimizers["our_resnet"].weight_decay = resnet_wd
            optimizers["our_head"].learning_rate = head_lr
            if hic_step:
                optimizers["our_hic"].learning_rate = hic_lr

            print(len(our_model.trainable_weights))
            print(len(our_model.get_layer("our_resnet").trainable_weights))
            print(len(our_model.get_layer("our_head").trainable_weights))
            if hic_step:
                print(len(our_model.get_layer("our_hic").trainable_weights))
    except Exception as e:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while compiling.")
        mp_q.put(None)
        return None

    try:
        our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE)
        # print(datetime.now().strftime('[%H:%M:%S] ') + "Saving " + str(current_epoch) + " model. ")
        # our_model.save(p.model_path + "_temp.h5", include_optimizer=False)
        # if os.path.exists(p.model_path):
        #     os.remove(p.model_path)
        # os.rename(p.model_path + "_temp.h5", p.model_path)
        safe_save(our_model.get_layer("our_resnet").get_weights(), p.model_path + "_res")
        safe_save(our_model.get_layer("our_head").get_weights(), p.model_path + "_head_" + head_name)
        safe_save(optimizers["our_resnet"].get_weights(), p.model_path + "_opt_resnet")
        safe_save(optimizers["our_head"].get_weights(), p.model_path + "_opt_head_" + head_name)
        if hic_step:
            safe_save(optimizers["our_hic"].get_weights(), p.model_path + "_opt_hic")
            safe_save(our_model.get_layer("our_hic").get_weights(), p.model_path + "_hic")
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        mp_q.put(None)
        return None

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch finished. ")
    mp_q.put(None)


def check_perf(mp_q, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    import tensorflow as tf
    import model as mo
    one_hot = joblib.load(f"pickle/{p.species[head_id]}_one_hot.gz")
    try:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.small_model(p.input_size, p.num_features, p.num_bins, len(heads[head_id]),
                                       p.bin_size)
            our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            our_model.get_layer("our_head").set_weights(joblib.load(p.model_path + "_head_hg38"))
        train_eval_chr = "chr1"
        train_eval_chr_info = []
        for info in train_info:
            if info[0] == train_eval_chr:
                train_eval_chr_info.append(info)
        print(f"Training set {len(train_eval_chr_info)}")
        training_spearman = evaluation.eval_perf(p, our_model, heads[head_id], train_eval_chr_info,
                                                 False, current_epoch, "train", one_hot, loaded_tracks)
        # training_spearman = 0
        print(f"Test set {len(test_info)}")
        test_spearman = evaluation.eval_perf(p, our_model, heads[head_id], test_info,
                                             True, current_epoch, "test", one_hot, loaded_tracks)
        with open(p.model_name + "_history.csv", "a+") as myfile:
            myfile.write(f"{training_spearman},{test_spearman}")
            myfile.write("\n")
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
    mp_q.put(None)


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def change_seq(x):
    return cm.rev_comp(x)


def load_data(mp_q, p, tracks, scores, t, t_end):
    scores_after_loading = np.zeros((len(scores), t_end-t, p.num_bins), dtype=np.float32)
    for i, track_name in enumerate(tracks):
        parsed_track = joblib.load(p.parsed_tracks_folder + track_name)
        for j in range(len(scores)):
            scores_after_loading[j, i] = parsed_track[scores[j, i, 0]][int(scores[j, i, 1]):int(scores[j, i, 2])].copy()
    joblib.dump(scores_after_loading, f"temp/data{t}", compress="lz4")
    mp_q.put(None)


last_proc = None
p = MainParams()
picked_training_regions = {}
if __name__ == '__main__':
    # import model as mo
    # our_model = mo.hic_model(p.input_size, p.num_features, p.num_bins, 50,  17, 190, 100)
    train_info, test_info, protein_coding = parser.parse_sequences(p.species, p.bin_size)
    if Path("pickle/track_names_col.gz").is_file():
        track_names_col = joblib.load("pickle/track_names_col.gz")
    else:
        track_names_col = parser.parse_tracks(p.species, p.bin_size, p.tracks_folder)

    for i in range(len(track_names_col)):
        print(f"Number of tracks in {p.species[i]}: {len(track_names_col[i])}")

    if Path("pickle/heads.gz").is_file():
        heads = joblib.load("pickle/heads.gz")
    else:
        heads = []
        for i in range(len(track_names_col)):
            new_head = random.sample(track_names_col[i], len(track_names_col[i]))
            new_head = [x for x in new_head if not x.startswith("sc")]
            heads.append(new_head)
        joblib.dump(heads, "pickle/heads.gz", compress="lz4")

    for i in range(len(heads)):
        print(f"Number of tracks in head {p.species[i]}: {len(heads[i])}")

    hic_keys = parser.parse_hic(p.parsed_hic_folder)
    hic_num = len(hic_keys)
    print(f"hic {hic_num}")

    mp_q = mp.Queue()
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

    print("Training starting")
    start_epoch = 0
    fit_epochs = 2
    try:
        for current_epoch in range(start_epoch, p.num_epochs, 1):
            if current_epoch % 2 == 0:
                head_id = 0
            else:
                head_id = 1 + (current_epoch - math.ceil(current_epoch / 2)) % (len(heads) - 1)

            # if current_epoch < 10:
            #     fit_epochs = 1
            # elif current_epoch < 40:
            #     fit_epochs = 2
            # else:
            #     fit_epochs = 3

            # check_perf(mp_q, 0)
            # exit()
            last_proc = run_epoch(last_proc, fit_epochs, head_id)
            if current_epoch % 100 == 0 and current_epoch != 0:  # and current_epoch != 0:
                print("Eval epoch")
                print(mp_q.get())
                last_proc.join()
                last_proc = None
                proc = mp.Process(target=check_perf, args=(mp_q, 0,))
                proc.start()
                print(mp_q.get())
                proc.join()
    except Exception as e:
        traceback.print_exc()

# for key in picked_training_regions.keys():
#     with open(f"dry/{key}_dry_test.bed", "w") as text_file:
#         text_file.write("\n".join(picked_training_regions[key]))
