import math
import os
import logging
import joblib
import gc
import random
import time
import pandas as pd
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
import tensorflow as tf
from scipy.ndimage import gaussian_filter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
matplotlib.use("agg")


def run_epoch(last_proc, fit_epochs, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " " + p.species[head_id])
    training_regions = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_regions.gz")
    one_hot = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_one_hot.gz")
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
        # Validation set!
        # if info[0] != "chr5" or not 139615843 < info[1] < 141668489: # or not 139615843 < info[1] < 141668489
        #     shifts.append(None)
        #     continue
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
        start_bin = int(info[1] / p.bin_size) - p.half_num_regions + shift_bins
        input_sequences.append(ns)
        output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
    if head_id == 0:
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["expression"], p)
        output_epigenome = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["epigenome"], p)
        output_conservation = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["conservation"], p)
    else:
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]], p)
    gc.collect()
    print("")
    half = len(input_sequences) // 2
    output_hic = []
    # print(f"Shifts {len(shifts)}")
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Hi-C")
    if head_id == 0:
        for hi, key in enumerate(hic_keys):
            # print(key, end=" ")
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
                hic_mat = gaussian_filter(hic_mat, sigma=1)
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
    input_sequences = np.asarray(input_sequences, dtype=bool)

    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr

    for i in range(half):
        output_expression[i] = np.flip(output_expression[i], axis=1)
        if head_id == 0:
            output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
            output_conservation[i] = np.flip(output_conservation[i], axis=1)

    all_outputs = {"our_expression": output_expression}
    if head_id == 0:
        all_outputs["our_epigenome"] = output_epigenome
        all_outputs["our_conservation"] = output_conservation
        all_outputs["our_hic"] = output_hic
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
    # joblib.dump([input_sequences, all_outputs], f"{p.pickle_folder}run.gz", compress=3)
    # argss = joblib.load(f"{p.pickle_folder}run.gz")
    # p = mp.Process(target=train_step, args=(argss[0][:400], argss[1][:400], argss[2][:400], fit_epochs, head_id,))
    proc = mp.Process(target=train_step, args=(
        heads[p.species[head_id]], p.species[head_id], input_sequences, all_outputs, fit_epochs, hic_num, mp_q,))
    proc.start()
    return proc


def safe_save(thing, place):
    joblib.dump(thing, place + "_temp", compress="lz4")
    if os.path.exists(place):
        os.remove(place)
    os.rename(place + "_temp", place)


def train_step(head, head_name, input_sequences, all_outputs, fit_epochs, hic_num, mp_q):
    human_training = True
    if head_name != "hg38" or hic_num == 0:
        human_training = False
    try:
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        # for device in physical_devices:
        #     tf.config.experimental.set_memory_growth(device, True)
        import model as mo
        if human_training:
            train_data = mo.wrap_for_human_training(input_sequences, all_outputs, p.GLOBAL_BATCH_SIZE)
            zero_fit_1 = np.zeros_like(input_sequences[0])
            zero_fit_2 = np.zeros_like(all_outputs["our_expression"][0])
            zero_fit_3 = np.zeros_like(all_outputs["our_epigenome"][0])
            zero_fit_4 = np.zeros_like(all_outputs["our_conservation"][0])
            zero_fit_5 = np.zeros_like(all_outputs["our_hic"][0])
            zero_data = mo.wrap_for_human_training([zero_fit_1],
                                                   {"our_expression": [zero_fit_2], "our_epigenome": [zero_fit_3],
                                                    "our_conservation": [zero_fit_4], "our_hic": [zero_fit_5]},
                                                   p.GLOBAL_BATCH_SIZE)
        else:
            train_data = mo.wrap(input_sequences, all_outputs["our_expression"], p.GLOBAL_BATCH_SIZE)
            zero_fit_1 = np.zeros_like(input_sequences[0])
            zero_fit_2 = np.zeros_like(all_outputs["our_expression"][0])
            zero_data = mo.wrap([zero_fit_1], [zero_fit_2], p.GLOBAL_BATCH_SIZE)

        del input_sequences
        del all_outputs
        gc.collect()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        # print(datetime.now().strftime('[%H:%M:%S] ') + "Loading the model")
        with strategy.scope():
            if human_training:
                our_model = mo.human_model(p.input_size, p.num_features, p.num_bins, hic_num, p.bin_size,p.hic_bin_size,
                                           head)
            else:
                our_model = mo.small_model(p.input_size, p.num_features, p.num_bins, len(head), p.bin_size)
            # Loading model weights
            if os.path.exists(p.model_path + "_res"):
                our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            if os.path.exists(p.model_path + "_expression_" + head_name):
                our_model.get_layer("our_expression").set_weights(
                    joblib.load(p.model_path + "_expression_" + head_name))
            if human_training and os.path.exists(p.model_path + "_epigenome"):
                our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            if human_training and os.path.exists(p.model_path + "_hic"):
                our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
            if human_training and os.path.exists(p.model_path + "_conservation"):
                our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
            print(f"=== Training with head {head_name} ===")

            # resnet_wd = 0.0000001
            # cap_e = min(current_epoch, 40)
            # transformer_lr = 0.0000005 + cap_e * 0.0000025
            # tfa.optimizers.AdamW
            # optimizers = [
            #     tf.keras.optimizers.Adam(learning_rate=resnet_lr),
            #     tfa.optimizers.AdamW(learning_rate=transformer_lr, weight_decay=transformer_wd),
            #     tf.keras.optimizers.Adam(learning_rate=expression_lr)
            # ]

            optimizers_and_layers = [(optimizers["our_resnet"], our_model.get_layer("our_resnet")),
                                     (optimizers["our_expression"], our_model.get_layer("our_expression"))
                                     ]
            if human_training:
                optimizers_and_layers.append((optimizers["our_hic"], our_model.get_layer("our_hic")))
                optimizers_and_layers.append((optimizers["our_epigenome"], our_model.get_layer("our_epigenome")))
                optimizers_and_layers.append((optimizers["our_conservation"], our_model.get_layer("our_conservation")))

            optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

            # our_model.get_layer("our_resnet").trainable = False

            if human_training:
                loss_weights = {}
                with open(str(p.script_folder) + "/../loss_weights") as f:
                    for line in f:
                        (key, val) = line.split()
                        loss_weights[key] = float(val)
                losses = {
                    "our_expression": "mse",
                    "our_epigenome": "mse",
                    "our_conservation": "mse",
                    "our_hic": "mse",
                }
                our_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
            else:
                our_model.compile(optimizer=optimizer, loss="mse")
            if current_epoch == start_epoch:
                our_model.fit(zero_data, steps_per_epoch=1, epochs=1)
                if os.path.exists(p.model_path + "_opt_resnet"):
                    print("loading resnet optimizer")
                    optimizers["our_resnet"].set_weights(joblib.load(p.model_path + "_opt_resnet"))

                if os.path.exists(p.model_path + "_opt_expression_" + head_name):
                    print("loading expression optimizer")
                    optimizers["our_expression"].set_weights(joblib.load(p.model_path + "_opt_expression_" + head_name))

                if human_training and os.path.exists(p.model_path + "_opt_hic"):
                    print("loading hic optimizer")
                    optimizers["our_hic"].set_weights(joblib.load(p.model_path + "_opt_hic"))

                if human_training and os.path.exists(p.model_path + "_opt_epigenome"):
                    print("loading epigenome optimizer")
                    optimizers["our_epigenome"].set_weights(joblib.load(p.model_path + "_opt_epigenome"))

                if human_training and os.path.exists(p.model_path + "_opt_conservation"):
                    print("loading conservation optimizer")
                    optimizers["our_conservation"].set_weights(joblib.load(p.model_path + "_opt_conservation"))

                optimizers["our_resnet"].learning_rate.assign(resnet_lr)
                optimizers["our_resnet"].weight_decay.assign(resnet_wd)
                optimizers["our_resnet"].clipnorm = resnet_clipnorm
                optimizers["our_expression"].learning_rate.assign(expression_lr)
                if human_training:
                    optimizers["our_hic"].learning_rate.assign(hic_lr)
                    optimizers["our_epigenome"].learning_rate.assign(epigenome_lr)
                    optimizers["our_conservation"].learning_rate.assign(conservation_lr)

    except Exception as e:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while compiling.")
        mp_q.put(None)
        return None

    try:
        our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE)
        safe_save(our_model.get_layer("our_resnet").get_weights(), p.model_path + "_res")
        safe_save(optimizers["our_resnet"].get_weights(), p.model_path + "_opt_resnet")
        safe_save(our_model.get_layer("our_expression").get_weights(), p.model_path + "_expression_" + head_name)
        safe_save(optimizers["our_expression"].get_weights(), p.model_path + "_opt_expression_" + head_name)
        if human_training:
            safe_save(optimizers["our_hic"].get_weights(), p.model_path + "_opt_hic")
            safe_save(our_model.get_layer("our_hic").get_weights(), p.model_path + "_hic")
            safe_save(optimizers["our_epigenome"].get_weights(), p.model_path + "_opt_epigenome")
            safe_save(our_model.get_layer("our_epigenome").get_weights(), p.model_path + "_epigenome")
            safe_save(optimizers["our_conservation"].get_weights(), p.model_path + "_opt_conservation")
            safe_save(our_model.get_layer("our_conservation").get_weights(), p.model_path + "_conservation")
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        mp_q.put(None)
        return None

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch finished. ")
    mp_q.put(None)
    return None


def check_perf(mp_q):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    import model as mo
    one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
    try:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.human_model(p.input_size, p.num_features, p.num_bins, hic_num, p.bin_size, p.hic_bin_size,
                                       heads["hg38"])
            our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
            our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
            our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
        train_eval_chr = "chr5"
        train_eval_chr_info = []
        for info in train_info:
            if info[0] == train_eval_chr and 139615843 < info[1] < 141668489:
                train_eval_chr_info.append(info)
        print(f"Training set {len(train_eval_chr_info)}")
        training_spearman = evaluation.eval_perf(p, our_model, heads["hg38"], train_eval_chr_info,
                                                 False, current_epoch, "train", one_hot)
        # training_spearman = 0
        test_eval_chr_info = []
        for info in test_info:
            # if info[0] == "chr11":
            test_eval_chr_info.append(info)
        print(f"Test set {len(test_eval_chr_info)}")
        test_spearman = evaluation.eval_perf(p, our_model, heads["hg38"], test_eval_chr_info,
                                             True, current_epoch, "test", one_hot)
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


last_proc = None
p = MainParams()
picked_training_regions = {}
if __name__ == '__main__':
    # import model as mo
    # import tensorflow as tf
    # nl = mo.pearsonr_poisson()
    #
    # x = np.random.random((2, 4, 4))
    # y = np.random.random((2, 4, 4))
    #
    # x = tf.convert_to_tensor(x, dtype=tf.float32)
    # y = tf.convert_to_tensor(y, dtype=tf.float32)
    # a = mo.loop_cor_loss(x, y)

    train_info, test_info, protein_coding = parser.parse_sequences(p)
    if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
        track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
    else:
        track_names_col = parser.parse_tracks(p)

    if Path(f"{p.pickle_folder}heads.gz").is_file():
        heads = joblib.load(f"{p.pickle_folder}heads.gz")
    else:
        heads = {}
        for specie in p.species:
            shuffled_tracks = random.sample(track_names_col[specie], len(track_names_col[specie]))
            if specie == "hg38":
                meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
                new_head = {}
                new_head["expression"] = [x for x in shuffled_tracks if
                                          meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                              ("CAGE", "RAMPAGE", "NETCAGE"))]
                new_head["epigenome"] = [x for x in shuffled_tracks if
                                         meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                             ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))]
                new_head["conservation"] = [x for x in shuffled_tracks if
                                            meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
                                                ("conservation"))]
            else:
                new_head = shuffled_tracks
                # new_head = [x for x in new_head if not x.startswith("sc")]
            heads[specie] = new_head
        joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress="lz4")

    for head_key in heads.keys():
        if head_key == "hg38":
            for human_key in heads[head_key]:
                print(f"Number of tracks in head {head_key} {human_key}: {len(heads[head_key][human_key])}")
        else:
            print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")

    hic_keys = parser.parse_hic(p)
    hic_num = len(hic_keys)
    print(f"hic {hic_num}")

    # import model as mo
    # our_model = mo.human_model(p.input_size, p.num_features, p.num_bins, 56, p.bin_size, p.hic_bin_size, heads["hg38"])

    hic_lr = 0.0001
    expression_lr = 0.0001
    conservation_lr = 0.0001
    epigenome_lr = 0.0001
    resnet_lr = 0.00001
    resnet_wd = 1e-7
    resnet_clipnorm = 0.001
    optimizers = {"our_resnet": tfa.optimizers.AdamW(learning_rate=resnet_lr, weight_decay=resnet_wd, clipnorm=resnet_clipnorm),
                  "our_expression": tf.keras.optimizers.Adam(learning_rate=expression_lr),
                  "our_epigenome": tf.keras.optimizers.Adam(learning_rate=epigenome_lr),
                  "our_conservation": tf.keras.optimizers.Adam(learning_rate=conservation_lr),
                  "our_hic": tf.keras.optimizers.Adam(learning_rate=hic_lr), }
    mp_q = mp.Queue()

    print("Training starting")
    start_epoch = 0
    fit_epochs = 20
    try:
        for current_epoch in range(start_epoch, p.num_epochs, 1):
            # if current_epoch % 2 == 0:
            #     head_id = 0
            # else:
            #     head_id = 1 + (current_epoch - math.ceil(current_epoch / 2)) % (len(heads) - 1)
            # Only human to test HIC and CON!
            head_id = 0
            if head_id == 0:
                p.STEPS_PER_EPOCH = 50
                fit_epochs = 3
            elif head_id == 1:
                p.STEPS_PER_EPOCH = 400
                fit_epochs = 2
            else:
                p.STEPS_PER_EPOCH = 1000
                fit_epochs = 1

            # check_perf(mp_q)
            # exit()
            last_proc = run_epoch(last_proc, fit_epochs, head_id)
            if current_epoch % 1000 == 0 and current_epoch != 0:  # and current_epoch != 0:
                print("Eval epoch")
                print(mp_q.get())
                last_proc.join()
                last_proc = None
                proc = mp.Process(target=check_perf, args=(mp_q,))
                proc.start()
                print(mp_q.get())
                proc.join()
    except Exception as e:
        traceback.print_exc()

# for key in picked_training_regions.keys():
#     with open(f"dry/{key}_dry_test.bed", "w") as text_file:
#         text_file.write("\n".join(picked_training_regions[key]))
