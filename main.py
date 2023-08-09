import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
import shutil
import math
import joblib
import gc
import random
import pandas as pd
import numpy as np
import common as cm
from pathlib import Path
import matplotlib
import psutil
import parse_data as parser
from datetime import datetime
import traceback
import multiprocessing as mp
from evaluation import evaluation
from main_params import MainParams
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')
import tensorflow_addons as tfa
import model as mo

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
matplotlib.use("agg")


def load_old_weights():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num if head_name == "hg38" else 0,
                              p.hic_size, heads)

    our_model_old = mo.make_model(p.input_size, p.num_features, p.num_bins, 6, p.hic_size, heads)
    our_model_old.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

    print("model loaded")
    for layer in our_model_old.get_layer("our_hic").layers:
        layer_name = layer.name
        layer_weights = layer.weights
        try:
            our_model.get_layer("our_hic").get_layer(layer_name).set_weights(layer_weights)
        except:
            try:
                nw = []
                for i, w in enumerate(our_model.get_layer("our_hic").get_layer(layer_name).get_weights()):
                    nw.append(np.resize(layer_weights[i], w.shape))
                our_model.get_layer("our_hic").get_layer(layer_name).set_weights(nw)
            except:
                print(layer_name)

    joblib.dump(our_model.get_layer("our_hic").get_weights(), p.model_folder + p.model_name + "_hic")
    print("dumped")


def get_train_data(num_seq, tss_only):
    input_sequences_all = []
    all_outputs = {}
    for head_key in heads.keys():
        for subkey in heads[head_key].keys():
            all_outputs[head_key + "_" + subkey] = []
        all_outputs[head_key + "_hic"] = []
    for specie in p.species:
        # training regions are shuffled each iteration
        if tss_only:
            shuffled_regions_info = data_split[specie]["train"]
            shuffled_regions_info = random.sample(shuffled_regions_info, len(shuffled_regions_info))
        else:
            min_len = min(len(training_regions[specie]), len(data_split[specie]["train"]))
            a = random.sample(training_regions[specie], min_len)
            b = random.sample(data_split[specie]["train"], min_len)
            shuffled_regions_info = a + b
            shuffled_regions_info = random.sample(shuffled_regions_info, len(shuffled_regions_info))
        input_sequences = []
        output_scores_info = []
        picked_regions = []
        for i, info in enumerate(shuffled_regions_info):
            if len(input_sequences) >= num_seq:
                break
            # Don't use chrY, chrM etc
            if info[0] not in one_hot[specie].keys():
                continue
            shift_bins = random.randint(-1 * (p.num_bins // 2), (p.num_bins // 2))
            pos_hic = info[1] + shift_bins * p.bin_size
            pos_hic = pos_hic - (pos_hic % p.hic_bin_size)
            start = pos_hic - (pos_hic % p.bin_size) - p.half_size
            extra = start + p.input_size - len(one_hot[specie][info[0]])
            if start < 0 or extra > 0:
                continue
            ns = one_hot[specie][info[0]][start:start + p.input_size]
            # less than 10% Ns
            if np.sum(ns[:, :-1]) < 0.9 * p.input_size:
                continue
            if specie in ["hg38", "mm10"]:
                if np.any(ns[:, -1]):
                    # Exclude region was encountered! Skipping
                    continue
            else:
                if np.any(one_hot[specie][info[0]][max(0, start - 131000):
                max(start + p.input_size + 131000, len(one_hot[specie][info[0]])), -1]):
                    # Exclude region was encountered! Skipping
                    continue
            dry_run_regions.append(f"{info[0]}\t{start}\t{start + p.input_size}\ttrain")
            picked_regions.append([info[0], pos_hic])
            start_bin = (start + p.half_size) // p.bin_size - p.half_num_regions
            input_sequences.append(ns)
            output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])

        print(datetime.now().strftime('[%H:%M:%S] ') + f"Loading parsed tracks [{specie}]")
        # half of sequences will be reverse-complemented
        half = len(input_sequences) // 2
        input_sequences = np.asarray(input_sequences, dtype=bool)
        # reverse-complement
        with mp.Pool(8) as pool:
            rc_arr = pool.map(cm.change_seq, input_sequences[:half])
        input_sequences[:half] = rc_arr
        # Cut off the test TSS layer
        input_sequences = input_sequences[:, :, :-1]
        print(input_sequences.shape)
        outputs = {}
        if specie == "hg38":
            outputs["conservation"] = parser.par_load_data(output_scores_info, heads[specie]["conservation"], p)
            outputs["expression_sc"] = parser.par_load_data(output_scores_info, heads[specie]["expression_sc"], p)
        outputs["expression"] = parser.par_load_data(output_scores_info, heads[specie]["expression"], p)
        outputs["epigenome"] = parser.par_load_data(output_scores_info, heads[specie]["epigenome"], p)
        # loading corresponding 2D data
        print(datetime.now().strftime('[%H:%M:%S] ') + f"Loading hic [{specie}]")
        outputs["hic"] = parser.par_load_hic_data(p.hic_keys[specie], p, picked_regions, half)
        print(datetime.now().strftime('[%H:%M:%S] ') + f"Done [{outputs['hic'].shape}]")
        # for reverse-complement sequences, the 1D output is flipped
        for i in range(half):
            outputs["expression"][i] = np.flip(outputs["expression"][i], axis=1)
            outputs["epigenome"][i] = np.flip(outputs["epigenome"][i], axis=1)
            if specie == "hg38":
                outputs["conservation"][i] = np.flip(outputs["conservation"][i], axis=1)
                outputs["expression_sc"][i] = np.flip(outputs["expression_sc"][i], axis=1)
        input_sequences_all.extend(input_sequences)
        # Copy the loaded data using zeroed data for the other species
        for head_key in heads.keys():
            for subkey in heads[head_key].keys():
                if len(p.species) > 1:
                    if specie == head_key:
                        all_outputs[head_key + "_" + subkey].append(outputs[subkey])
                    else:
                        all_outputs[head_key + "_" + subkey].append(np.zeros((num_seq, len(heads[head_key][subkey]), p.num_bins), dtype=np.float16))
                else:
                    all_outputs[head_key + "_" + subkey] = outputs[subkey]
            if len(p.species) > 1:
                if specie == head_key:
                    all_outputs[head_key + "_hic"].append(outputs["hic"])
                else:
                    all_outputs[head_key + "_hic"].append(np.zeros((num_seq, len(p.hic_keys[head_key]), p.hic_size), dtype=np.float16))
            else:
                all_outputs[head_key + "_hic"] = outputs["hic"]
    if len(p.species) > 1:
        for key in all_outputs.keys():
            all_outputs[key] = np.concatenate(all_outputs[key], axis=0, dtype=np.float16)
    gc.collect()
    print(datetime.now().strftime('[%H:%M:%S] ') + "Returning loaded data")
    print_memory()
    joblib.dump((input_sequences_all, all_outputs), f"{p.temp_folder}data.p")


def make_model(heads):
    with strategy.scope():
        our_model, head_inds = mo.make_model(p.input_size, p.num_features, p.num_bins, p.hic_keys, p.hic_size, heads)
        optimizers = {}
        optimizers_and_layers = []
        for k in ["stem", "body", "3d_projection", "hg38_conservation", "hg38_expression_sc"]:
            if k in ["stem", "body"]:
                optimizers[k] = tfa.optimizers.AdamW(learning_rate=p.lrs[k], weight_decay=p.wds[k],
                                                     epsilon=1e-08, clipnorm=0.01)
            else:
                optimizers[k] = tfa.optimizers.AdamW(learning_rate=p.lrs[k], weight_decay=p.wds[k],
                                                     epsilon=1e-08)
            optimizers_and_layers.append((optimizers[k], our_model.get_layer(k)))
        losses = {}
        for specie in p.species:
            for k in ["expression", "epigenome", "hic"]:
                optimizers[specie + "_" + k] = tfa.optimizers.AdamW(
                    learning_rate=p.lrs[k], weight_decay=p.wds[k], epsilon=1e-08)
                optimizers_and_layers.append((optimizers[specie + "_" + k], our_model.get_layer(specie + "_" + k)))
        
        for specie in p.species:
            for k in ["expression", "epigenome"]:
                losses[specie + "_" + k] = mo.skip_poisson
            losses[specie + "_hic"] = mo.skip_mse
        losses["hg38_conservation"] = mo.skip_mse
        losses["hg38_expression_sc"] = mo.skip_poisson
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        if os.path.exists(p.model_folder + p.model_name + "_loss_scale"):
            initial_scale = joblib.load(p.model_folder + p.model_name + "_loss_scale") // 4
            print(f"Initial scale: {initial_scale}")
        else:
            initial_scale = 1
        opt_scaled = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=initial_scale,
                                                                 dynamic_growth_steps=20)
        our_model.compile(optimizer=opt_scaled, loss=losses, loss_weights=p.loss_weights)
        
    return our_model, head_inds, optimizers, opt_scaled


def check_perf():
    try:
        training_result = {}
        valid_result = {}
        hist = ""
        for test_sp in p.species:
            print(test_sp)
            inds = []
            for key in heads[test_sp]:
                inds.append(head_inds[test_sp + "_" + key])
            train_info = data_split[test_sp]["train"]
            valid_info = data_split[test_sp]["valid"]
            train_eval_info = random.sample(train_info, len(train_info) // 10)
            print(f"Training set {len(train_eval_info)}")
            training_result[test_sp] = evaluation.eval_perf(p, our_model, heads[test_sp], train_eval_info,
                                                   False, current_epoch, "train", one_hot[test_sp], inds)
            # training_result = "0"
            valid_eval_chr_info = []
            for info in valid_info:
                # if info[0] == "chr2":
                valid_eval_chr_info.append(info)
            # valid_eval_chr_info = random.sample(valid_eval_chr_info, len(train_info) // 10)
            print(f"Valid set {len(valid_eval_chr_info)}")
            valid_result[test_sp] = evaluation.eval_perf(p, our_model, heads[test_sp], valid_eval_chr_info,
                                                False, current_epoch, "valid", one_hot[test_sp], inds)
            hist += training_result[test_sp] + "\t" + valid_result[test_sp]

        with open(p.model_name + "_history.tsv", "a+") as myfile:
            myfile.write(hist + "\n")
        new_folder = p.model_folder + valid_result["hg38"] + "/"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        save_weights(new_folder)
        target_dir = os.path.join(script_dir, valid_result["hg38"])
        os.makedirs(target_dir)
        for py_file in py_files.keys():
            with open(os.path.join(target_dir, py_file), 'w') as file:
                file.write(py_files[py_file])
    except Exception as e:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def change_seq(x):
    return cm.rev_comp(x)


def save_weights(target_folder=None):
    if target_folder is None:
        target_folder = p.model_folder
    Path(p.model_folder + "temp/").mkdir(parents=True, exist_ok=True)
    files = []
    for k in ["stem", "body", "3d_projection", "hg38_conservation", "hg38_expression_sc"]:
        joblib.dump(our_model.get_layer(k).get_weights(),
                    cm.store(p.model_folder + "temp/" + p.model_name + "_" + k, files))
        joblib.dump(optimizers[k].get_weights(),  cm.store(p.model_folder + "temp/" + p.model_name + "_opt_" + k, files))
    for specie in p.species:
        for k in ["expression", "epigenome", "hic"]:
            joblib.dump(our_model.get_layer(specie + "_" + k).get_weights(),
                        cm.store(p.model_folder + "temp/" + p.model_name + "_" + specie + "_" + k, files))
            joblib.dump(optimizers[specie + "_" + k].get_weights(),
                        cm.store(p.model_folder + "temp/" + p.model_name + "_opt_" + specie + "_" + k, files))
    joblib.dump(opt_scaled.loss_scale.numpy(), cm.store(p.model_folder + "temp/" + p.model_name + "_loss_scale", files))
    for file_name in files:
        shutil.copy(file_name, target_folder + os.path.basename(file_name))
    print(datetime.now().strftime('[%H:%M:%S] ') + "Weights saved.")


def train_one_epoch():
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading saved data")
    saved_train_data = joblib.load(f"{p.temp_folder}data.p")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Data loaded")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Wrapping the data")
    train_data = mo.wrap_for_training(saved_train_data[0], saved_train_data[1], p.GLOBAL_BATCH_SIZE)
    del saved_train_data
    print(datetime.now().strftime('[%H:%M:%S] ') + "Data wrapped")
    try:
        if current_epoch == 0 and os.path.exists(p.model_path + "_opt_stem"):
            # need to init the optimizers
            print(datetime.now().strftime('[%H:%M:%S] ') + "Optimizers init")
            mo.load_weights(p, our_model)
            our_model.fit(train_data, steps_per_epoch=1, epochs=1)
            mo.load_weights(p, our_model)
            # loading the previous optimizer weights
            for k in ["stem", "body", "3d_projection", "hg38_conservation", "hg38_expression_sc"]:
                if os.path.exists(p.model_path + "_opt_" + k):
                    print(f"Loading optimizer {k}")
                    optimizers[k].set_weights(joblib.load(p.model_path + "_opt_" + k))
            for specie in p.species:
                for k in ["expression", "epigenome", "hic"]:
                    if os.path.exists(p.model_path + "_opt_" + specie + "_" + k):
                        print(f"Loading optimizer {specie}_{k}")
                        optimizers[specie + "_" + k].set_weights(joblib.load(p.model_path + "_opt_" + specie + "_" + k))

        from tensorflow.keras.callbacks import TerminateOnNaN
        term = TerminateOnNaN()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Model fit ...")
        history = our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE,
                                callbacks=[term])  # verbose=0
        if np.isnan(history.history['loss']).any():
            print("Exiting because of nan")
            exit()
        print(datetime.now().strftime('[%H:%M:%S] ') + f"Final loss {history.history['loss'][-1]}")

        del history
        del term
        del train_data
        gc.collect()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " finished.")
    except:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")


if __name__ == '__main__':
    make_data_proc = None
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(script_dir)
    py_files = {}
    for py_file in ['model.py', 'main.py', 'parse_data.py', 'main_params.py']:
        with open(os.path.join(script_dir, py_file), 'r') as file:
            py_files[py_file] = file.read()
    p = MainParams()
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    dry_run_regions = []
    data_split = parser.parse_sequences(p)
    if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
        track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
    else:
        track_names_col = parser.parse_tracks(p)

    meta = pd.read_csv("data/all_track.metadata.tsv", sep="\t")
    if Path(f"{p.pickle_folder}heads.gz").is_file():
        heads = joblib.load(f"{p.pickle_folder}heads.gz")
    else:
        heads = {}
        for specie in p.species:
            shuffled_tracks = random.sample(track_names_col[specie], len(track_names_col[specie]))
            if specie in ["hg38", "mm10"]:
                shuffled_tracks = [x for x in shuffled_tracks if not x.startswith(("scAtlas", "scATAC", "scEnd5"))]
                meta_filenames = set(meta['file_name'])
                shuffled_tracks = [track for track in shuffled_tracks if track in meta_filenames]
                new_head = {"expression": [x for x in shuffled_tracks if
                                           meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                               ("CAGE", "NETCAGE", "RAMPAGE"))],
                            "epigenome": [x for x in shuffled_tracks if
                                          meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                              ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))]
                            }
                if specie == "hg38":
                    new_head["conservation"] = [x for x in shuffled_tracks if
                                                meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
                                                    "conservation")]
                    new_head["expression_sc"] = parser.parse_tracks_sc(p)
            else:
                new_head = shuffled_tracks
            heads[specie] = new_head
        for head_key in heads.keys():
            for key2 in heads[head_key].keys():
                print(f"Number of tracks in head {head_key} {key2}: {len(heads[head_key][key2])}")
        joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress=3)
    training_regions = {}
    one_hot = {}
    for specie in p.species:
        training_regions[specie] = joblib.load(f"{p.pickle_folder}{specie}_regions.gz")
        one_hot[specie] = joblib.load(f"{p.pickle_folder}{specie}_one_hot.gz")

    print("Training starting")
    our_model, head_inds, optimizers, opt_scaled = make_model(heads)
    num_seq = p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH
    mo.load_weights(p, our_model)
    start_epoch = 0
    fit_epochs = 1

    make_data_proc = mp.Process(target=get_train_data, args=(num_seq, p.tss_only,))
    make_data_proc.start()
    make_data_proc.join()
    make_data_proc.close()

    for current_epoch in range(start_epoch, p.num_epochs, 1):
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
        try:
            make_data_proc = mp.Process(target=get_train_data, args=(num_seq, p.tss_only,))
            make_data_proc.start()

            train_one_epoch()

            make_data_proc.join()
            make_data_proc.close()

            if current_epoch % 5 == 0:
                save_weights()
            if current_epoch % 100 == 0 and current_epoch != 0:
                print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating.")
                check_perf()
                with open(f"dry_test.bed", "a+") as text_file:
                    text_file.write("\n".join(dry_run_regions))
        except Exception as e:
            print(f"Problem with the epoch {current_epoch}!")
            print(e)
            traceback.print_exc()
            make_data_proc = None
