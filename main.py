import math
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
    import model as mo
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, heads)

    our_model_old = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, heads, True)
    our_model_old.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))

    print("model loaded")
    for layer in our_model_old.get_layer("our_resnet").layers:
        layer_name = layer.name
        layer_weights = layer.weights
        try:
            our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(layer_weights)
        except:
            print(layer_name)

    joblib.dump(our_model.get_layer("our_resnet").get_weights(), p.model_folder + p.model_name + "_res")
    print("dumped")


def get_data_and_train(fit_epochs, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
    training_regions = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_regions.gz")
    one_hot = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_one_hot.gz")
    # training regions are shuffled each iteration
    if head_id == 0:
        shuffled_regions_info = training_regions + train_info
        shuffled_regions_info = random.sample(shuffled_regions_info, len(shuffled_regions_info))
    else:
        shuffled_regions_info = random.sample(training_regions, len(training_regions))
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
        # No Ns
        if np.sum(ns[:, :-1]) != p.input_size:
            continue
        if head_id == 0:
            if np.any(ns[:, -1]):
                # Exclude region was encountered! Skipping
                continue
        else:
            if np.any(one_hot[info[0]][max(0, start - 131000):max(start + p.input_size + 131000, len(one_hot[info[0]])),
                      -1]):
                # Exclude region was encountered! Skipping
                continue
        dry_run_regions.append(f"{info[0]}\t{start}\t{start + p.input_size}\ttrain")
        picked_regions.append([info[0], pos_hic])
        start_bin = (start + p.half_size) // p.bin_size - p.half_num_regions
        input_sequences.append(ns)
        output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])

    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
    # half of sequences will be reverse-complemented
    half = len(input_sequences) // 2
    if head_id == 0:
        output_conservation = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["conservation"], p)
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["expression"], p)
        output_epigenome = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["epigenome"], p)
        # loading corresponding 2D data
        output_hic = parser.par_load_hic_data(hic_keys, p, picked_regions, half)
    else:
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]], p)
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
        if p.species[head_id] == "hg38":
            output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
            output_conservation[i] = np.flip(output_conservation[i], axis=1)

    all_outputs = {"our_expression": output_expression}
    if p.species[head_id] == "hg38":
        all_outputs["our_epigenome"] = output_epigenome
        all_outputs["our_conservation"] = output_conservation
        if hic_num > 0:
            all_outputs["our_hic"] = output_hic
    # Cut off the test TSS layer
    input_sequences = input_sequences[:, :, :-1]
    print(input_sequences.shape)
    print_memory()
    if head_id == 0:
        train_data = mo.wrap_for_human_training(input_sequences, all_outputs, p.GLOBAL_BATCH_SIZE)
    else:
        train_data = mo.wrap(input_sequences, all_outputs["our_expression"], p.GLOBAL_BATCH_SIZE)
    del input_sequences
    del all_outputs
    gc.collect()
    print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
    train(train_data, p.species[head_id])
    del train_data
    gc.collect()


def make_model(heads, head_name):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num if head_name == "hg38" else 0,
                                  p.hic_size, heads)
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
                if key not in ["our_stem", "our_body"]:
                    loss_weights[key] = float(weight)
                learning_rates[key] = float(lr)
                weight_decays[key] = float(wd)
        optimizers = {}
        optimizers["our_stem"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_stem"],
                                                      weight_decay=weight_decays["our_stem"], epsilon=1e-08,
                                                      clipnorm=0.01)
        optimizers["our_body"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_body"],
                                                      weight_decay=weight_decays["our_body"], epsilon=1e-08,
                                                      clipnorm=0.01)
        for specie in p.species:
            optimizers["our_expression_" + specie] = tfa.optimizers.AdamW(
                learning_rate=learning_rates["our_expression"], weight_decay=weight_decays["our_expression"],
                epsilon=1e-08)
        optimizers["our_epigenome"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_epigenome"],
                                                           weight_decay=weight_decays["our_epigenome"], epsilon=1e-08)
        optimizers["our_conservation"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_conservation"],
                                                              weight_decay=weight_decays["our_conservation"],
                                                              epsilon=1e-08)
        optimizers["our_hic"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_hic"],
                                                     weight_decay=weight_decays["our_hic"], epsilon=1e-08)

        optimizers_and_layers = [(optimizers["our_stem"], our_model.get_layer("our_stem")),
                                 (optimizers["our_body"], our_model.get_layer("our_body")),
                                 (optimizers["our_expression_" + head_name], our_model.get_layer("our_expression"))]
        if head_name == "hg38":
            if hic_num > 0:
                optimizers_and_layers.append((optimizers["our_hic"], our_model.get_layer("our_hic")))
            optimizers_and_layers.append((optimizers["our_epigenome"], our_model.get_layer("our_epigenome")))
            optimizers_and_layers.append((optimizers["our_conservation"], our_model.get_layer("our_conservation")))
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        if head_name == "hg38":
            losses = {
                "our_expression": "poisson",
                "our_epigenome": "poisson",
                "our_conservation": "mse",
            }
            if hic_num > 0:
                losses["our_hic"] = "mse"
        else:
            losses = "poisson"
        if os.path.exists(p.model_folder + "loss_scale_" + head_name):
            initial_scale = joblib.load(p.model_folder + "loss_scale_" + head_name)
            print(f"Initial scale: {initial_scale}")
        else:
            initial_scale = 1
        opt_scaled = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=initial_scale,
                                                                 dynamic_growth_steps=10)
        if head_name == "hg38":
            our_model.compile(optimizer=opt_scaled, loss=losses, loss_weights=loss_weights)
        else:
            our_model.compile(optimizer=opt_scaled, loss=losses)
        # loading model weights
        if os.path.exists(p.model_path + "_stem"):
            our_model.get_layer("our_stem").set_weights(joblib.load(p.model_path + "_stem"))
        if os.path.exists(p.model_path + "_body"):
            our_model.get_layer("our_body").set_weights(joblib.load(p.model_path + "_body"))
        if os.path.exists(p.model_path + "_expression_" + head_name):
            our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_" + head_name))
        if head_name == "hg38":
            if os.path.exists(p.model_path + "_epigenome"):
                our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            if hic_num > 0 and os.path.exists(p.model_path + "_hic"):
                our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
            if os.path.exists(p.model_path + "_conservation"):
                our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
    return our_model, optimizers, opt_scaled


def train(train_data, head_name):
    try:
        if current_epoch == 0 and os.path.exists(p.model_path + "_opt_resnet"):
            # need to init the optimizers
            ls = opt_scaled.loss_scale.numpy()
            opt_scaled.loss_scale.assign(1)
            our_model.fit(train_data, steps_per_epoch=1, epochs=1)
            opt_scaled.loss_scale.assign(ls)
            # loading the previous optimizer weights
            if os.path.exists(p.model_path + "_opt_stem"):
                print("loading stem optimizer")
                optimizers["our_stem"].set_weights(joblib.load(p.model_path + "_opt_stem"))
            if os.path.exists(p.model_path + "_opt_body"):
                print("loading body optimizer")
                optimizers["our_body"].set_weights(joblib.load(p.model_path + "_opt_body"))
            if os.path.exists(p.model_path + "_opt_expression_" + head_name):
                print("loading expression optimizer")
                optimizers["our_expression_" + head_name].set_weights(
                    joblib.load(p.model_path + "_opt_expression_" + head_name))
            if head_name == "hg38":
                if hic_num > 0 and os.path.exists(p.model_path + "_opt_hic"):
                    print("loading hic optimizer")
                    optimizers["our_hic"].set_weights(joblib.load(p.model_path + "_opt_hic"))
                if os.path.exists(p.model_path + "_opt_epigenome"):
                    print("loading epigenome optimizer")
                    optimizers["our_epigenome"].set_weights(joblib.load(p.model_path + "_opt_epigenome"))
                if os.path.exists(p.model_path + "_opt_conservation"):
                    print("loading conservation optimizer")
                    optimizers["our_conservation"].set_weights(joblib.load(p.model_path + "_opt_conservation"))

        from tensorflow.keras.callbacks import TerminateOnNaN
        term = TerminateOnNaN()
        history = our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE, callbacks=[term])
        if np.isnan(history.history['loss']).any():
            print("Skipping because of nan")
            opt_scaled.loss_scale.assign(max(1, opt_scaled.loss_scale.numpy() // 2))
        del history
        del term
        gc.collect()
    except:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        opt_scaled.loss_scale.assign(max(1, opt_scaled.loss_scale.numpy() // 2))


def check_perf():
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    try:
        one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
        auc = 0  # get_linking_AUC()
        train_eval_info = random.sample(train_info, len(train_info) // 10)
        print(f"Training set {len(train_eval_info)}")
        training_result = evaluation.eval_perf(p, our_model, heads["hg38"], train_eval_info,
                                               False, current_epoch, "train", one_hot)
        # training_result = "0"
        valid_eval_chr_info = []
        for info in valid_info:
            # if info[0] == "chr2":
            valid_eval_chr_info.append(info)
        print(f"Valid set {len(valid_eval_chr_info)}")
        valid_result = evaluation.eval_perf(p, our_model, heads["hg38"], valid_eval_chr_info,
                                            False, current_epoch, "valid", one_hot)
        with open(p.model_name + "_history.tsv", "a+") as myfile:
            myfile.write(training_result + "\t" + valid_result + "\t" + str(auc) + "\n")
        new_folder = p.model_folder + valid_result + "_" + str(auc) + "/"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        file_names = os.listdir(p.model_folder + "temp/")
        for file_name in file_names:
            shutil.copy(p.model_folder + "temp/" + file_name, new_folder + file_name)
    except Exception as e:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")


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
    if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
        track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
    else:
        track_names_col = parser.parse_tracks(p)

    heads = joblib.load(f"{p.pickle_folder}heads.gz")
    for head_key in heads.keys():
        if head_key == "hg38":
            for human_key in heads[head_key]:
                print(f"Number of tracks in head {head_key} {human_key}: {len(heads[head_key][human_key])}")
        else:
            print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")

    hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    # hic_keys = []
    hic_num = len(hic_keys)
    print(f"hic {hic_num}")

    print("Training starting")
    # load_old_weights()
    # exit()
    our_model, optimizers, opt_scaled = make_model(heads[p.species[0]], p.species[0])
    start_epoch = 0
    fit_epochs = 1
    for current_epoch in range(start_epoch, p.num_epochs, 1):
        if current_epoch % 2 == 0:
            head_id = 0
        else:
            head_id = 1 + (current_epoch - math.ceil(current_epoch / 2)) % (len(heads) - 1)
        head_id = 0
        head_name = p.species[0]
        # check_perf()
        # exit()
        get_data_and_train(fit_epochs, head_id)
        if current_epoch % 1 == 0:
            Path(p.model_folder + "temp/").mkdir(parents=True, exist_ok=True)
            joblib.dump(our_model.get_layer("our_stem").get_weights(),
                        p.model_folder + "temp/" + p.model_name + "_stem")
            joblib.dump(our_model.get_layer("our_body").get_weights(),
                        p.model_folder + "temp/" + p.model_name + "_body")
            joblib.dump(optimizers["our_body"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_body")
            joblib.dump(optimizers["our_stem"].get_weights(), p.model_folder + "temp/" + p.model_name + "_opt_stem")
            joblib.dump(our_model.get_layer("our_expression").get_weights(),
                        p.model_folder + "temp/" + p.model_name + "_expression_" + head_name)
            joblib.dump(optimizers["our_expression_" + head_name].get_weights(),
                        p.model_folder + "temp/" + p.model_name + "_opt_expression_" + head_name)
            if head_name == "hg38":
                if hic_num > 0:
                    joblib.dump(optimizers["our_hic"].get_weights(),
                                p.model_folder + "temp/" + p.model_name + "_opt_hic")
                    joblib.dump(our_model.get_layer("our_hic").get_weights(),
                                p.model_folder + "temp/" + p.model_name + "_hic")
                joblib.dump(optimizers["our_epigenome"].get_weights(),
                            p.model_folder + "temp/" + p.model_name + "_opt_epigenome")
                joblib.dump(our_model.get_layer("our_epigenome").get_weights(),
                            p.model_folder + "temp/" + p.model_name + "_epigenome")
                joblib.dump(optimizers["our_conservation"].get_weights(),
                            p.model_folder + "temp/" + p.model_name + "_opt_conservation")
                joblib.dump(our_model.get_layer("our_conservation").get_weights(),
                            p.model_folder + "temp/" + p.model_name + "_conservation")
            joblib.dump(opt_scaled.loss_scale.numpy(), p.model_folder + "temp/" + "loss_scale_" + head_name)
            file_names = os.listdir(p.model_folder + "temp/")
            for file_name in file_names:
                shutil.copy(p.model_folder + "temp/" + file_name, p.model_folder + file_name)
            joblib.dump(opt_scaled.loss_scale.numpy(), p.model_folder + "loss_scale")
        if current_epoch % 15 == 0 and current_epoch != 0:  # and current_epoch != 0:
            print("Eval epoch")
            check_perf()

with open(f"dry_test.bed", "a+") as text_file:
    text_file.write("\n".join(dry_run_regions))