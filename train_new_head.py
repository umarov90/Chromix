import os
import shutil
import joblib
import gc
import random
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
from evaluation import evaluation
from main_params import MainParams
matplotlib.use("agg")


def get_data_and_train(last_proc, fit_epochs):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
    # training regions are shuffled each iteration
    shuffled_regions_info = random.sample(training_regions, len(training_regions_tss)) + training_regions_tss
    shuffled_regions_info = random.sample(shuffled_regions_info, len(shuffled_regions_info))
    input_sequences = []
    output_scores_info = []
    for i, info in enumerate(shuffled_regions_info):
        if len(input_sequences) >= p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH:
            break
        # Don't use chrY, chrM etc
        if info[0] not in one_hot.keys():
            continue
        shift_bins = random.randint(-1 * (p.num_bins // 2), (p.num_bins // 2))
        pos = info[1] + shift_bins * p.bin_size
        start = pos - (pos % p.bin_size) - p.half_size
        extra = start + p.input_size - len(one_hot[info[0]])
        if start < 0 or extra > 0:
            continue
        ns = one_hot[info[0]][start:start + p.input_size]
        if np.any(ns[:, -1]):
            # Exclude region was encountered! Skipping
            continue

        start_bin = (start + p.half_size) // p.bin_size - p.half_num_regions
        input_sequences.append(ns)
        output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])

    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
    output_expression = parser.par_load_data(output_scores_info, track_names, p)
    # half of sequences will be reverse-complemented
    half = len(input_sequences) // 2
    input_sequences = np.asarray(input_sequences, dtype=bool)
    # reverse-complement
    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr
    # for reverse-complement sequences, the 1D output is flipped
    for i in range(half):
        output_expression[i] = np.flip(output_expression[i], axis=1)
    # Cut off the exclude layer
    input_sequences = input_sequences[:, :, :-1]

    if last_proc is not None:
        print("Waiting")
        print(mp_q.get())
        last_proc.join()
        print("Finished waiting")
    proc = mp.Process(target=make_model_and_train, args=(track_names, input_sequences, output_expression, fit_epochs, mp_q,))
    proc.start()
    return proc


def make_model_and_train(track_names, input_sequences, output_expression, fit_epochs, mp_q):
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    import tensorflow_addons as tfa
    import model as mo
    try:
        train_data = mo.wrap_for_human_training(input_sequences, output_expression, p.GLOBAL_BATCH_SIZE)
        del input_sequences
        del output_expression
        gc.collect()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, track_names)
            optimizers_and_layers = [(optimizers["our_resnet"], our_model.get_layer("our_resnet")),
                                     (optimizers["our_expression"], our_model.get_layer("our_expression"))]
            frozen_epochs = 10
            frozen = current_epoch < frozen_epochs
            if frozen:
                our_model.get_layer("our_resnet").trainable = False
            print(our_model.summary())
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
            if os.path.exists(p.model_folder + "loss_scale"):
                initial_scale = joblib.load(p.model_folder + "loss_scale")
                print(f"Initial scale: {initial_scale}")
            else:
                initial_scale = 1
            opt_scaled = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=initial_scale, dynamic_growth_steps=200)
            our_model.compile(optimizer=opt_scaled, loss="poisson")
            if (current_epoch == 0 or current_epoch == frozen_epochs) and os.path.exists(p.model_path + "_opt_resnet"):
                # need to init the optimizers
                # to prevent bug with fp16 loss being too big for initial scaling
                if os.path.exists(p.model_path + "_res"):
                    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
                if os.path.exists(p.model_path + "_expression"):
                    our_model.get_layer("our_expression").set_weights(
                        joblib.load(p.model_path + "_expression"))
                # Will be reloaded later
                our_model.fit(train_data, steps_per_epoch=1, epochs=1)
                # loading the previous optimizer weights
                if os.path.exists(p.model_path + "_opt_expression"):
                    print("loading expression optimizer")
                    optimizers["our_expression"].set_weights(joblib.load(p.model_path + "_opt_expression"))
                    optimizers["our_expression"].learning_rate.assign(learning_rates["our_expression"])
                    optimizers["our_expression"].weight_decay.assign(weight_decays["our_expression"])
                if not frozen and os.path.exists(p.model_path + "_opt_resnet"):
                    print("loading resnet optimizer")
                    optimizers["our_resnet"].set_weights(joblib.load(p.model_path + "_opt_resnet"))
                    optimizers["our_resnet"].learning_rate.assign(learning_rates["our_resnet"])
                    optimizers["our_resnet"].weight_decay.assign(weight_decays["our_resnet"])
            # loading model weights
            if os.path.exists(p.model_path + "_res"):
                our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            if os.path.exists(p.model_path + "_expression"):
                our_model.get_layer("our_expression").set_weights(
                    joblib.load(p.model_path + "_expression"))
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
        if not frozen:
            joblib.dump(our_model.get_layer("our_resnet").get_weights(), p.model_folder + "temp/" + p.model_name + "_res", compress="lz4")
            joblib.dump(optimizers["our_resnet"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_resnet", compress="lz4")
        joblib.dump(our_model.get_layer("our_expression").get_weights(), p.model_folder + "temp/" + p.model_name + "_expression", compress="lz4")
        joblib.dump(optimizers["our_expression"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_expression", compress="lz4")
        file_names = os.listdir(p.model_folder + "temp/")
        for file_name in file_names:
            shutil.copy(p.model_folder + "temp/" + file_name, p.model_folder + file_name)
        joblib.dump(opt_scaled.loss_scale.numpy(), p.model_folder + "loss_scale")
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
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        import model as mo
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads)
            our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression"))
        train_eval_info = random.sample(train_info, len(train_info) // 10)
        print(f"Training set {len(train_eval_info)}")
        training_result = evaluation.eval_perf(p, our_model, heads, train_eval_info,
                                               False, current_epoch, "train", one_hot)
        # training_result = "0"
        valid_eval_chr_info = []
        for info in valid_info:
            # if info[0] == "chr2":
            valid_eval_chr_info.append(info)
        print(f"Valid set {len(valid_eval_chr_info)}")
        valid_result = evaluation.eval_perf(p, our_model, heads, valid_eval_chr_info,
                                            False, current_epoch, "valid", one_hot)
        with open(p.model_name + "_history.tsv", "a+") as myfile:
            myfile.write(training_result + "\t" + valid_result + "\t" + str(auc) + "\n")
        new_folder = p.model_folder + valid_result + "_" + str(auc) + "/"
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
    track_names = parser.parse_new_tracks(p)
    print(f"Number of tracks: {len(track_names)}")
    import tensorflow_addons as tfa
    loss_weights = {}
    learning_rates = {}
    weight_decays = {}
    with open(str(p.script_folder) + "/../loss_weights") as f:
        for line in f:
            if line.startswith("#"):
                continue
            (key, weight, lr, wd) = line.split()
            if hic_num == 0 and key == "our_hic":
                continue
            if key != "our_resnet":
                loss_weights[key] = float(weight)
            learning_rates[key] = float(lr)
            weight_decays[key] = float(wd)
    optimizers = {}
    optimizers["our_resnet"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_resnet"], weight_decay=weight_decays["our_resnet"])
    optimizers["our_expression"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_expression"], weight_decay=weight_decays["our_expression"])

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
            if current_epoch % 50 == 0 and current_epoch != 0:  # and current_epoch != 0:
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