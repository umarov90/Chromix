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
import pathlib
import pickle
import evaluation
from main_params import MainParams
import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
matplotlib.use("agg")


def create_model(q):
    import tensorflow as tf
    import model as mo
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = mo.small_model(p.input_size, p.num_features, p.num_regions, p.out_stack_num)
        # our_model = mo.hic_model(input_size, num_features, num_regions, out_stack_num, hic_num, hic_size)
        # print("loading model")
        # our_model_old = tf.keras.models.load_model(model_folder + "bigger_transformer.h5",
        #                                            custom_objects={'PatchEncoder': mo.PatchEncoder})
        # print("model loaded")
        # for layer in our_model_old.get_layer("our_resnet").layers:
        #     layer_name = layer.name
        #     layer_weights = layer.weights
        #     our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(layer_weights)
        #
        # for layer in our_model_old.get_layer("our_transformer").layers:
        #     layer_name = layer.name
        #     layer_weights = layer.weights
        #     if "input" in layer_name:
        #         continue
        #     if isinstance(layer, mo.PatchEncoder):
        #         layer_weights = layer.projection.weights
        #         our_model.get_layer("our_transformer").get_layer(layer_name).projection.set_weights(layer_weights)
        #     elif isinstance(layer.weights, list):
        #         try:
        #             new_weights_list = []
        #             for li in range(len(layer.weights)):
        #                 new_shape = our_model.get_layer("our_transformer").get_layer(layer_name).weights[li].shape
        #                 new_weights_list.append(np.resize(layer.weights[li], new_shape))
        #             our_model.get_layer("our_transformer").get_layer(layer_name).set_weights(new_weights_list)
        #         except:
        #             pass
        #     else:
        #         new_shape = our_model.get_layer("our_transformer").get_layer(layer_name).shape
        #         our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(np.resize(layer_weights, new_shape))

        # for layer in our_model_old.get_layer("our_head").layers:
        #     layer_name = layer.name
        #     if "input" in layer_name:
        #         continue
        #     layer_weights = layer.weights
        #     our_model.get_layer("our_head").get_layer(layer_name).set_weights(layer_weights)

        # for layer in our_model_old.get_layer("our_hic").layers:
        #     layer_name = layer.name
        #     if "input" in layer_name:
        #         continue
        #     layer_weights = layer.weights
        #     our_model.get_layer("our_hic").get_layer(layer_name).set_weights(layer_weights)

        # our_model.set_weights(joblib.load(model_folder + model_name + "_w"))

        our_model.save(p.model_path, include_optimizer=False)
        print("Model saved " + p.model_path)
        for head_id in range(len(heads)):
            joblib.dump(our_model.get_layer("our_head").get_weights(),
                        p.model_path + "_head_" + str(head_id), compress=3)
    q.put(None)


def run_epoch(last_proc, fit_epochs, head_id):
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
    shuffled_info = random.sample(train_info, len(train_info))
    input_sequences = []
    output_scores = []
    shifts = []
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
    err = 0
    for i, info in enumerate(shuffled_info):
        if len(input_sequences) >= p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH:
            break
        if i % 500 == 0:
            print(i, end=" ")
            gc.collect()
        try:
            shift_bins = random.randint(-int(current_epoch / p.shift_speed) - p.initial_shift, int(current_epoch / p.shift_speed) + p.initial_shift)
            start = info[1] - (info[1] % p.bin_size) - p.half_size + shift_bins * p.bin_size
            extra = start + p.input_size - len(one_hot[info[0]])
            if start < 0 or extra > 0:
                shifts.append(None)
                continue
            shifts.append(shift_bins)
            if start < 0:
                ns = one_hot[info[0]][0:start + p.input_size]
                ns = np.concatenate((np.zeros((-1 * start, p.num_features)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, p.num_features))))
            else:
                ns = one_hot[info[0]][start:start + p.input_size]
            start_bin = int(info[1] / p.bin_size) - p.half_num_regions + shift_bins
            scores = []
            for key in heads[head_id]:
                scores.append([info[0], start_bin, start_bin + p.num_regions])
                # scores.append(gas[key][info[0]][start_bin: start_bin + num_regions])
            input_sequences.append(ns)
            output_scores.append(scores)
        except Exception as e:
            print(e)
            err += 1
    # print("")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")

    gc.disable()
    start_time = time.time()
    for i, key in enumerate(heads[head_id]):
        # if i % 1000 == 0:
        #     loaded_tracks = joblib.load(parsed_blocks_folder + str(int(i / 1000) + 1))
        if i % 1000 == 0 and i != 0:
            end_time = time.time()
            print(f"{i} ({(end_time - start_time):.2f})", end=" ")
            start_time = time.time()
            gc.collect()
        if key in loaded_tracks.keys():
            # buf = loaded_tracks[key]
            # parsed_track = joblib.load(buf)
            # buf.seek(0)
            parsed_track = loaded_tracks[key]
        else:
            parsed_track = joblib.load(p.parsed_tracks_folder + key)
            # with open(p.parsed_tracks_folder + key, 'rb') as fp:
            #     parsed_track = pickle.load(fp)

        for s in output_scores:
            s[i] = parsed_track[s[i][0]][s[i][1]:s[i][2]].copy()
        # with Pool(4) as p:
        #     map_arr = p.starmap(load_values, zip(output_scores, repeat( [i, parsed_track] )))
    gc.enable()
    # print("")
    # print(np.asarray(output_scores).shape)
    half = len(input_sequences) // 2
    output_hic = []
    # print(f"Shifts {len(shifts)}")
    # print(datetime.now().strftime('[%H:%M:%S] ') + "Hi-C")
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
            start_row = hd['locus1'].searchsorted(start_hic - p.hic_bin_size, side='left')
            end_row = hd['locus1'].searchsorted(end_hic, side='right')
            hd = hd.iloc[start_row:end_row]
            # convert start of the input region to the bin number
            start_hic = int(start_hic / p.hic_bin_size)
            # subtract start bin from the binned entries in the range [start_row : end_row]
            l1 = (np.floor(hd["locus1"].values / p.hic_bin_size) - start_hic).astype(int)
            l2 = (np.floor(hd["locus2"].values / p.hic_bin_size) - start_hic).astype(int)
            hic_score = hd["score"].values
            # drop contacts with regions outside the [start_row : end_row] range
            lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
            l1 = l1[lix]
            l2 = l2[lix]
            hic_score = hic_score[lix]
            hic_mat[l1, l2] += hic_score
            hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
            # hic_mat = gaussian_filter(hic_mat, sigma=1)
            if ni < half:
                hic_mat = np.rot90(hic_mat, k=2)
            if i == 0:
                print(f"original {hic_mat.shape}")
            hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
            if i == 0:
                print(f"triu {hic_mat.shape}")
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
    output_scores = np.asarray(output_scores)
    if np.isnan(output_scores).any() or np.isinf(output_scores).any():
        print("nan in the output")
        exit()
    input_sequences = np.asarray(input_sequences, dtype=bool)

    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr

    for i in range(half):
        output_scores[i] = np.flip(output_scores[i], axis=1)

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
    proc = mp.Process(target=train_step, args=(input_sequences, output_scores, output_hic, fit_epochs, head_id,))
    proc.start()
    return proc


def safe_save(thing, place):
    joblib.dump(thing, place + "_temp", compress=3)
    if os.path.exists(place):
        os.remove(place)
    os.rename(place + "_temp", place)


def train_step(input_sequences, output_scores, output_hic, fit_epochs, head_id):
    try:
        import tensorflow as tf
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        import model as mo
        if len(output_hic) > 0:
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
            our_model = tf.keras.models.load_model(p.model_path,
                                                   custom_objects={'PatchEncoder': mo.PatchEncoder})
            our_model.get_layer("our_head").set_weights(joblib.load(p.model_path + "_head_" + str(head_id)))
            print(f"=== Training with head {head_id} ===")
            hic_lr = 0.0001
            head_lr = 0.001
            head_wd = 0.00001
            transformer_lr = 0.0001
            transformer_wd = 0.00001
            resnet_lr = 0.00001
            resnet_wd = 0.000001
            # cap_e = min(current_epoch, 10)
            # resnet_lr = 0.00001 + cap_e * 0.00001
            # tfa.optimizers.AdamW
            # optimizers = [
            #     tf.keras.optimizers.Adam(learning_rate=resnet_lr),
            #     tfa.optimizers.AdamW(learning_rate=transformer_lr, weight_decay=transformer_wd),
            #     tf.keras.optimizers.Adam(learning_rate=head_lr)
            # ]
            optimizers = [
                tf.keras.optimizers.Adam(learning_rate=resnet_lr),
                tf.keras.optimizers.Adam(learning_rate=transformer_lr),
                tf.keras.optimizers.Adam(learning_rate=head_lr)
            ]

            optimizers_and_layers = [(optimizers[0], our_model.get_layer("our_resnet")),
                                     (optimizers[1], our_model.get_layer("our_transformer")),
                                     (optimizers[2], our_model.get_layer("our_head"))
                                     ]
            if hic_num > 0:
                optimizers.append(tfa.optimizers.AdamW(learning_rate=hic_lr, weight_decay=0.0001))
                optimizers_and_layers.append((optimizers[3], our_model.get_layer("our_hic")))

            optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

            our_model.get_layer("our_transformer").trainable = True
            our_model.get_layer("our_resnet").trainable = True

            if hic_num > 0:
                loss_weights = {"our_head": 1.0, "our_hic": 0.02}
                losses = {
                    "our_head": "mse",
                    "our_hic": "mse",
                }
                our_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
            else:
                our_model.compile(optimizer=optimizer, loss="mse")

            our_model.fit(zero_data, steps_per_epoch=1, epochs=1)

            if os.path.exists(p.model_path + "_w"):
                our_model.set_weights(joblib.load(p.model_path + "_w"))

            if os.path.exists(p.model_path + "_opt_resnet"):
                print("loading resnet optimizer")
                optimizers[0].set_weights(joblib.load(p.model_path + "_opt_resnet"))

            if os.path.exists(p.model_path + "_opt_transformer"):
                print("loading transformer optimizer")
                optimizers[1].set_weights(joblib.load(p.model_path + "_opt_transformer"))

            if os.path.exists(p.model_path + "_opt_head_" + str(head_id)):
                print("loading head optimizer")
                optimizers[2].set_weights(joblib.load(p.model_path + "_opt_head_" + str(head_id)))

            if hic_num > 0 and os.path.exists(p.model_path + "_opt_hic"):
                print("loading hic optimizer")
                optimizers[3].set_weights(joblib.load(p.model_path + "_opt_hic"))

            # optimizers[0].epsilon = 1e-8
            # optimizers[1].epsilon = 1e-8
            # optimizers[2].epsilon = 1e-8
            # optimizers[0].learning_rate = resnet_lr
            # optimizers[1].learning_rate = transformer_lr
            # optimizers[2].learning_rate = head_lr
            # if hic_num > 0:
            #     optimizers[3].learning_rate = hic_lr

        print(len(our_model.trainable_weights))
        print(len(our_model.get_layer("our_transformer").trainable_weights))
        print(len(our_model.get_layer("our_resnet").trainable_weights))
        print(len(our_model.get_layer("our_head").trainable_weights))
        if hic_num > 0:
            print(len(our_model.get_layer("our_hic").trainable_weights))
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while compiling.")
        mp_q.put(None)
        return None

    try:
        our_model.fit(train_data, epochs=fit_epochs)
        # print(datetime.now().strftime('[%H:%M:%S] ') + "Saving " + str(current_epoch) + " model. ")
        our_model.save(p.model_path + "_temp.h5", include_optimizer=False)
        if os.path.exists(p.model_path):
            os.remove(p.model_path)
        os.rename(p.model_path + "_temp.h5", p.model_path)
        safe_save(our_model.get_layer("our_head").get_weights(), p.model_path + "_head_" + str(head_id))
        safe_save(optimizers[0].get_weights(), p.model_path + "_opt_resnet")
        safe_save(optimizers[1].get_weights(), p.model_path + "_opt_transformer")
        safe_save(optimizers[2].get_weights(), p.model_path + "_opt_head_" + str(head_id))
        if hic_num > 1:
            safe_save(optimizers[3].get_weights(), p.model_path + "_opt_hic")
        joblib.dump(our_model.get_weights(), p.model_path + "_w", compress=3)
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        mp_q.put(None)
        return None

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " finished. ")
    mp_q.put(None)


def check_perf(mp_q, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    import tensorflow as tf
    import model as mo
    try:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = tf.keras.models.load_model(p.model_path,
                                                   custom_objects={'PatchEncoder': mo.PatchEncoder})
            our_model.get_layer("our_head").set_weights(
                joblib.load(p.model_path + "_head_" + str(head_id)))
        train_eval_chr = "chr2"
        train_eval_chr_info = []
        train_info_eval = joblib.load("pickle/train_info_eval.gz")
        for info in train_info_eval:
            if info[0] == train_eval_chr:
                train_eval_chr_info.append(info)
        train_eval_chr_info.sort(key=lambda x: x[1])
        print(f"Training set {len(train_eval_chr_info)}")
        training_spearman = evaluation.eval_perf(p, our_model,  heads[head_id], train_eval_chr_info,
                                                 False, current_epoch, train_eval_chr, one_hot, hic_keys, loaded_tracks)
        test_info_eval = joblib.load("pickle/test_info_eval.gz")
        print(f"Test set {len(test_info_eval)}")
        test_info_eval.sort(key=lambda x: x[1])
        test_spearman = evaluation.eval_perf(p, our_model,  heads[head_id], test_info_eval,
                                             False, current_epoch, "chr1", one_hot, hic_keys, loaded_tracks)
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
if __name__ == '__main__':
    # import model as mo
    # our_model = mo.small_model(input_size, num_features, num_regions, out_stack_num)
    script_folder = pathlib.Path(__file__).parent.resolve()
    folders = open(str(script_folder) + "/data_dirs").read().strip().split("\n")
    os.chdir(folders[0])
    p.parsed_tracks_folder = folders[1]
    p.parsed_hic_folder = folders[2]
    p.model_folder = folders[3]

    # draw = joblib.load("draw")
    # viz.draw_tracks(*draw)

    ga, one_hot, train_info, test_info, tss_loc, protein_coding = parser.get_sequences(p.bin_size, p.chromosomes)
    parser.parse_eval_data(p.chromosomes)
    if Path("pickle/track_names.gz").is_file():
        track_names = joblib.load("pickle/track_names.gz")
    else:
        track_names = parser.parse_tracks(ga, p.bin_size, tss_loc, p.chromosomes, p.tracks_folder)
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
    # heads = [heads[0][:5000]]
    # joblib.dump(heads, "pickle/heads.gz", compress=3)
    # hic_keys = parser.parse_hic()
    # hic_keys = ["hic_THP1_10kb_interactions.txt.bz2"]  # , "hic_A549_10kb_interactions.txt.bz2"
    # "hic_HepG2_10kb_interactions.txt.bz2", "hic_THP1_10kb_interactions.txt.bz2"]
    hic_keys = []
    hic_num = len(hic_keys)

    mp_q = mp.Queue()

    if not Path(p.model_folder + p.model_name).is_file():
        proc = mp.Process(target=create_model, args=(mp_q,))
        proc.start()
        print(mp_q.get())
        proc.join()
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

    print("Training starting")
    start_epoch = 0
    fit_epochs = 1
    for current_epoch in range(start_epoch, p.num_epochs, 1):
        head_id = current_epoch % len(heads)
        if current_epoch < 10:
            fit_epochs = 1
        elif current_epoch < 40:
            fit_epochs = 2
        elif current_epoch < 80:
            fit_epochs = 4
        else:
            fit_epochs = 8

        check_perf(mp_q, 0)
        exit()
        last_proc = run_epoch(last_proc, fit_epochs, head_id)
        if current_epoch % 10 == 0 and current_epoch >= 20: # and current_epoch != 0:
            print("Eval epoch")
            print(mp_q.get())
            last_proc.join()
            last_proc = None
            proc = mp.Process(target=check_perf, args=(mp_q, 0,))
            proc.start()
            print(mp_q.get())
            proc.join()
