import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
import shutil
import math
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
from scipy.ndimage import gaussian_filter
import cooler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
matplotlib.use("agg")


def get_data_and_train(last_proc, fit_epochs):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
    # training regions are shuffled each iteration
    shuffled_regions_info = random.sample(training_regions, len(training_regions_tss)) + training_regions_tss
    shuffled_regions_info = random.sample(shuffled_regions_info, len(shuffled_regions_info))
    input_sequences = []
    output_scores_info = []
    picked_regions = []
    for i, info in enumerate(shuffled_regions_info):
        if len(input_sequences) >= p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH:
            break
        # Don't use chrY, chrM etc
        if info[0] not in one_hot.keys():
            continue
        shift_bins = random.randint(-1 * (p.num_bins // 2), (p.num_bins // 2))
        pos_hic = info[1] + shift_bins * p.bin_size
        pos_hic = pos_hic - (pos_hic % p.hic_bin_size)
        start = pos_hic - (pos_hic % p.bin_size) - p.half_size
        extra = start + p.input_size - len(one_hot[info[0]])
        if start < 0 or extra > 0:
            continue
        ns = one_hot[info[0]][start:start + p.input_size]
        if np.any(ns[:, -1]):
            # Exclude region was encountered! Skipping
            continue

        dry_run_regions.append(f"{info[0]}\t{start}\t{start + p.input_size}\ttrain")
        picked_regions.append([info[0], pos_hic])
        start_bin = (start + p.half_size) // p.bin_size - p.half_num_regions
        input_sequences.append(ns)
        output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])

    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
    output_conservation = parser.par_load_data(output_scores_info, heads["conservation"], p)
    output_expression = parser.par_load_data(output_scores_info, heads["expression"], p)
    output_sc = parser.par_load_data(output_scores_info, heads["sc"], p)
    output_epigenome = parser.par_load_data(output_scores_info, heads["epigenome"], p)
    # half of sequences will be reverse-complemented
    half = len(input_sequences) // 2
    # loading corresponding 2D data
    output_hic = parser.par_load_hic_data(hic_keys, p, picked_regions, half)
    gc.collect()
    print_memory()
    input_sequences = np.asarray(input_sequences, dtype=bool)

    # reverse-complement
    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr

    # for reverse-complement sequences, the 1D output is flipped
    for i in range(half):
        output_expression[i] = np.flip(output_expression[i], axis=1)
        output_sc[i] = np.flip(output_sc[i], axis=1)
        output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
        output_conservation[i] = np.flip(output_conservation[i], axis=1)

    all_outputs = {"our_expression": output_expression}
    all_outputs["our_epigenome"] = output_epigenome
    all_outputs["our_conservation"] = output_conservation
    all_outputs["our_sc"] = output_sc
    if hic_num > 0:
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
    proc = mp.Process(target=make_model_and_train, args=(heads, input_sequences, all_outputs, fit_epochs, hic_num, mp_q,))
    proc.start()
    return proc


def make_model_and_train(heads, input_sequences, all_outputs, fit_epochs, hic_num, mp_q):
    import tensorflow as tf
    import tensorflow_addons as tfa
    import model as mo
    try:
        train_data = mo.wrap_for_human_training(input_sequences, all_outputs, p.GLOBAL_BATCH_SIZE)
        del input_sequences
        del all_outputs
        gc.collect()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, heads)
            optimizers_and_layers = [(optimizers["our_resnet"], our_model.get_layer("our_resnet")),
                                     (optimizers["our_expression"], our_model.get_layer("our_expression"))]
            if hic_num > 0:
                optimizers_and_layers.append((optimizers["our_hic"], our_model.get_layer("our_hic")))
            optimizers_and_layers.append((optimizers["our_epigenome"], our_model.get_layer("our_epigenome")))
            optimizers_and_layers.append((optimizers["our_conservation"], our_model.get_layer("our_conservation")))
            optimizers_and_layers.append((optimizers["our_sc"], our_model.get_layer("our_sc")))
            optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
            print("Loss updated")
            # loading the loss weights
            with open(str(p.script_folder) + "/../loss_weights") as f:
                for line in f:
                    (key, weight, lr, wd) = line.split()
                    if hic_num == 0 and key == "our_hic":
                        continue
                    if key != "our_resnet":
                        loss_weights[key] = float(weight)
                    learning_rates[key] = float(lr)
                    weight_decays[key] = float(wd)
            print(optimizers["our_resnet"].weight_decay)
            optimizers["our_expression"].learning_rate.assign(learning_rates["our_expression"])
            optimizers["our_expression"].weight_decay.assign(weight_decays["our_expression"])
            optimizers["our_resnet"].learning_rate.assign(learning_rates["our_resnet"])  
            optimizers["our_resnet"].weight_decay.assign(weight_decays["our_resnet"])          
            optimizers["our_epigenome"].learning_rate.assign(learning_rates["our_epigenome"])
            optimizers["our_epigenome"].weight_decay.assign(weight_decays["our_epigenome"])       
            optimizers["our_conservation"].learning_rate.assign(learning_rates["our_conservation"])
            optimizers["our_conservation"].weight_decay.assign(weight_decays["our_conservation"]) 
            print(loss_weights["our_sc"])  
            if hic_num > 0:
                optimizers["our_hic"].learning_rate.assign(learning_rates["our_hic"])
                optimizers["our_hic"].weight_decay.assign(weight_decays["our_hic"])   
            losses = {
                    "our_expression": mo.fast_mse1,
                    "our_sc": mo.fast_mse5,
                    "our_epigenome": mo.fast_mse01,
                    "our_conservation":  "mse",
                }
            if hic_num > 0:
                losses["our_hic"] = mo.fast_mse01
            our_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
            if current_epoch == 0 and os.path.exists(p.model_path + "_opt_resnet"):
                # need to init the optimizers
                our_model.fit(train_data, steps_per_epoch=1, epochs=1)
                # loading the previous optimizer weights
                if os.path.exists(p.model_path + "_opt_expression"):
                    print("loading expression optimizer")
                    optimizers["our_expression"].set_weights(joblib.load(p.model_path + "_opt_expression"))
                if os.path.exists(p.model_path + "_opt_resnet"):
                    print("loading resnet optimizer")
                    optimizers["our_resnet"].set_weights(joblib.load(p.model_path + "_opt_resnet"))
    
                if hic_num > 0 and os.path.exists(p.model_path + "_opt_hic"):
                    print("loading hic optimizer")
                    optimizers["our_hic"].set_weights(joblib.load(p.model_path + "_opt_hic"))

                if os.path.exists(p.model_path + "_opt_epigenome"):
                    print("loading epigenome optimizer")
                    optimizers["our_epigenome"].set_weights(joblib.load(p.model_path + "_opt_epigenome"))

                if os.path.exists(p.model_path + "_opt_conservation"):
                    print("loading conservation optimizer")
                    optimizers["our_conservation"].set_weights(joblib.load(p.model_path + "_opt_conservation"))
                if os.path.exists(p.model_path + "_opt_sc"):
                    print("loading sc optimizer")
                    optimizers["our_sc"].set_weights(joblib.load(p.model_path + "_opt_sc"))
            # loading model weights
            if os.path.exists(p.model_path + "_res"):
                our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            if os.path.exists(p.model_path + "_expression"):
                our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
            if os.path.exists(p.model_path + "_epigenome"):
                our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            if hic_num > 0 and os.path.exists(p.model_path + "_hic"):
                our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
            if os.path.exists(p.model_path + "_conservation"):
                our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
            if os.path.exists(p.model_path + "_sc"):
                our_model.get_layer("our_sc").set_weights(joblib.load(p.model_path + "_sc"))
    except:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while compiling.")
        mp_q.put(None)
        return None

    try:
        our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE)
        del train_data
        gc.collect()
        Path(p.model_folder + "temp/").mkdir(parents=True, exist_ok=True)
        joblib.dump(our_model.get_layer("our_resnet").get_weights(), p.model_folder + "temp/" + p.model_name + "_res", compress="lz4")
        joblib.dump(optimizers["our_resnet"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_resnet", compress="lz4")
        joblib.dump(our_model.get_layer("our_expression").get_weights(), p.model_folder + "temp/" + p.model_name + "_expression", compress="lz4")
        joblib.dump(optimizers["our_expression"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_expression", compress="lz4")
        joblib.dump(our_model.get_layer("our_sc").get_weights(), p.model_folder + "temp/" + p.model_name + "_sc", compress="lz4")
        joblib.dump(optimizers["our_sc"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_sc", compress="lz4")
        if hic_num > 0:
            joblib.dump(optimizers["our_hic"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_hic", compress="lz4")
            joblib.dump(our_model.get_layer("our_hic").get_weights(), p.model_folder + "temp/" + p.model_name + "_hic", compress="lz4")
        joblib.dump(optimizers["our_epigenome"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_epigenome", compress="lz4")
        joblib.dump(our_model.get_layer("our_epigenome").get_weights(), p.model_folder + "temp/" + p.model_name + "_epigenome", compress="lz4")
        joblib.dump(optimizers["our_conservation"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_conservation", compress="lz4")
        joblib.dump(our_model.get_layer("our_conservation").get_weights(), p.model_folder + "temp/" + p.model_name + "_conservation", compress="lz4")
        file_names = os.listdir(p.model_folder + "temp/")
        for file_name in file_names:
            shutil.copy(p.model_folder + "temp/" + file_name, p.model_folder + file_name)
    except:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        mp_q.put(None)
        return None

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch finished. ")
    mp_q.put(None)
    return None


def check_perf(mp_q):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    import tensorflow as tf
    import model as mo
    try:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads)
            our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
            our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
            our_model.get_layer("our_sc").set_weights(joblib.load(p.model_path + "_sc"))
        train_eval_chr = "chr1"
        train_eval_chr_info = []
        for info in train_info:
            if info[0] == train_eval_chr:
                train_eval_chr_info.append(info)
        print(f"Training set {len(train_eval_chr_info)}")
        # training_result = evaluation.eval_perf(p, our_model, heads, train_eval_chr_info,
        #                                          False, current_epoch, "train", one_hot)
        training_result = "0"
        valid_eval_chr_info = []
        for info in valid_info:
            if info[0] == "chr2":
                valid_eval_chr_info.append(info)
        print(f"Valid set {len(valid_eval_chr_info)}")
        valid_result = evaluation.eval_perf(p, our_model, heads, valid_eval_chr_info,
                                             False, current_epoch, "valid", one_hot)
        with open(p.model_name + "_history.tsv", "a+") as myfile:
            myfile.write(training_result + "\t" + valid_result + "\n")
        new_folder = p.model_folder + valid_result + "/"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        file_names = os.listdir(p.model_folder)
        for file_name in file_names:
            if file_name.startswith(p.model_name) and os.path.isfile(os.path.join(p.model_folder, file_name)):
                shutil.copy(p.model_folder + file_name, new_folder + file_name)
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
dry_run_regions = []
if __name__ == '__main__':
    train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
    track_names = parser.parse_tracks(p)
    meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
    if Path(f"{p.pickle_folder}heads.gz").is_file():
        heads = joblib.load(f"{p.pickle_folder}heads.gz")
    else:
        heads = {}
        heads["sc"] = [x for x in track_names if x.startswith(("scATAC", "scEnd5"))]
        track_names = [x for x in track_names if not x.startswith(("scATAC", "scEnd5"))]
        meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
        heads["expression"] = [x for x in track_names if
                                   meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                       ("CAGE", "RAMPAGE", "NETCAGE"))]

        heads["epigenome"] = [x for x in track_names if
                                  meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                      ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))]
        heads["conservation"] = [x for x in track_names if
                                     meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
                                         "conservation")]
        joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress="lz4")

    for key in heads.keys():
        print(f"Number of tracks in head {key}: {len(heads[key])}")

    hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    # hic_keys = []
    hic_num = len(hic_keys)
    print(f"hic {hic_num}")

    import tensorflow_addons as tfa
    loss_weights = {}
    learning_rates = {}
    weight_decays = {}
    with open(str(p.script_folder) + "/../loss_weights") as f:
        for line in f:
            (key, weight, lr, wd) = line.split()
            if hic_num == 0 and key == "our_hic":
                continue
            if key != "our_resnet":
                loss_weights[key] = float(weight)
            learning_rates[key] = float(lr)
            weight_decays[key] = float(wd)
    optimizers = {}
    optimizers["our_resnet"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_resnet"], weight_decay=weight_decays["our_resnet"], clipnorm=0.001)
    optimizers["our_expression"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_expression"], weight_decay=weight_decays["our_expression"], clipnorm=0.001)
    optimizers["our_sc"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_sc"], weight_decay=weight_decays["our_sc"], clipnorm=0.001)
    optimizers["our_epigenome"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_epigenome"], weight_decay=weight_decays["our_epigenome"], clipnorm=0.001)
    optimizers["our_conservation"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_conservation"], weight_decay=weight_decays["our_conservation"], clipnorm=0.001)
    optimizers["our_hic"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_hic"], weight_decay=weight_decays["our_hic"], clipnorm=0.001)

    training_regions = joblib.load(f"{p.pickle_folder}regions.gz")
    training_regions_tss = joblib.load(f"{p.pickle_folder}train_info.gz")
    one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")

    mp_q = mp.Queue()
    print("Training starting")
    start_epoch = 0
    fit_epochs = 1
    try:
        for current_epoch in range(start_epoch, p.num_epochs, 1):
            # check_perf(mp_q)
            # exit()
            last_proc = get_data_and_train(last_proc, fit_epochs)
            if current_epoch % 10 == 0 and current_epoch != 0:  # and current_epoch != 0:
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

with open(f"dry_test.bed", "a+") as text_file:
    text_file.write("\n".join(dry_run_regions))
