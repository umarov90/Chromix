import os
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
import model as mo
import torch
from torch.utils.data import DataLoader
from torch import autocast, nn
from torch.cuda.amp import GradScaler
matplotlib.use("agg")


def get_train_data(q, head_id):
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
        # less than 70% Ns
        if np.sum(ns[:, :-1]) < 0.7 * p.input_size:
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
    input_sequences = np.asarray(input_sequences, dtype=bool)
    # reverse-complement
    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr
    # Cut off the test TSS layer
    input_sequences = input_sequences[:, :, :-1]
    print(input_sequences.shape)

    if head_id == 0:
        output_conservation = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["conservation"], p)
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["expression"], p)
        output_epigenome = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["epigenome"], p)
        # loading corresponding 2D data
        output_hic = parser.par_load_hic_data(p.hic_keys, p, picked_regions, half)
    else:
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]], p)
    print_memory()
    # for reverse-complement sequences, the 1D output is flipped
    for i in range(half):
        output_expression[i] = np.flip(output_expression[i], axis=1)
        if p.species[head_id] == "hg38":
            output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
            output_conservation[i] = np.flip(output_conservation[i], axis=1)

    all_outputs = {"expression": output_expression}
    if p.species[head_id] == "hg38":
        all_outputs["epigenome"] = output_epigenome
        all_outputs["conservation"] = output_conservation
        if p.hic_num > 0:
            all_outputs["hic"] = output_hic
    print_memory()
    if head_name == "hg38":
        q.put((input_sequences, all_outputs))
    else:
        q.put((input_sequences, all_outputs["expression"]))


def make_model(p):
    model = mo.Chromix(p).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model = nn.DataParallel(model)
    return model, optimizer


def train(dataloader, head_name):
    # torch.cuda.mem_get_info(device=None)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X = X.to(device)
        yd = {}
        for key in y.keys():
            yd[key] = y[key].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            # Compute prediction error
            loss_hm = model(X, target=yd, head="human")
            # loss_ms = model(X, target=y, head="mouse")
            total_loss = loss_hm.mean()
            # total_loss = loss_hm + loss_ms
        scaler.scale(total_loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        if batch % 2 == 0:
            loss, current = total_loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def check_perf():
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    try:
        one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
        auc = 0  # get_linking_AUC()
        train_eval_info = random.sample(train_info, len(train_info) // 100)
        print(f"Training set {len(train_eval_info)}")
        training_result = evaluation.eval_perf(p, model, device, heads["hg38"], train_eval_info,
                                               False, current_epoch, "train", one_hot)
        # training_result = "0"
        valid_eval_chr_info = []
        for info in valid_info:
            # if info[0] == "chr2":
            valid_eval_chr_info.append(info)
        print(f"Valid set {len(valid_eval_chr_info)}")
        valid_result = evaluation.eval_perf(p, model, device, heads["hg38"], valid_eval_chr_info,
                                            False, current_epoch, "valid", one_hot)
        with open(p.model_name + "_history.tsv", "a+") as myfile:
            myfile.write(training_result + "\t" + valid_result + "\t" + str(auc) + "\n")
        new_folder = p.model_folder + valid_result + "_" + str(auc) + "/"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        file_names = os.listdir(p.model_folder + "temp/")
        for file_name in file_names:
            shutil.copy(p.model_folder + "temp/" + file_name, new_folder + file_name)
        target_dir = os.path.join(script_dir, valid_result)
        os.makedirs(target_dir)
        shutil.copy(os.path.join(script_dir, 'model.py'), target_dir)
        shutil.copy(os.path.join(script_dir, 'main.py'), target_dir)
        shutil.copy(os.path.join(script_dir, 'parse_data.py'), target_dir)
    except Exception as e:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def change_seq(x):
    return cm.rev_comp(x)


def load_weights():
    if os.path.exists(p.model_folder + p.model_name):
        checkpoint = torch.load(p.model_folder + p.model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    else:
        return 0



make_data_proc = None
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)
p = MainParams()
dry_run_regions = []
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    train_info, valid_info, test_info, protein_coding = parser.parse_sequences(p)
    if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
        track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
    else:
        track_names_col = parser.parse_tracks(p, train_info + valid_info + test_info)

    if Path(f"{p.pickle_folder}heads.gz").is_file():
        heads = joblib.load(f"{p.pickle_folder}heads.gz")
    else:
        heads = {}
        for specie in p.species:
            shuffled_tracks = random.sample(track_names_col[specie], len(track_names_col[specie]))
            if specie == "hg38":
                sc_head = [x for x in shuffled_tracks if x.startswith(("scAtlas", "scATAC"))]
                shuffled_tracks = [x for x in shuffled_tracks if not x.startswith(("scAtlas", "scATAC", "scEnd5"))]
                # joblib.dump(sc_head, f"{p.pickle_folder}sc_head.gz", compress="lz4")
                meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
                new_head = {"expression": [x for x in shuffled_tracks if
                                           meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                               ("CAGE", "NETCAGE"))],
                            "epigenome": [x for x in shuffled_tracks if
                                          meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                              ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))],
                            "conservation": [x for x in shuffled_tracks if
                                             meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
                                                 "conservation")]}
            else:
                new_head = shuffled_tracks
            heads[specie] = new_head
        joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress="lz4")
    for head_key in heads.keys():
        if isinstance(heads[head_key], dict):
            for key2 in heads[head_key].keys():
                print(f"Number of tracks in head {head_key} {key2}: {len(heads[head_key][key2])}")
                p.output_heads[head_key + "_" + key2] = len(heads[head_key][key2])
        else:
            print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")
            p.output_heads[head_key + "_expression"] = len(heads[head_key])

    
    print("Training starting")
    model, optimizer = make_model(p)
    scaler = GradScaler()
    start_epoch = load_weights()
    fit_epochs = 1
    mp_q = mp.Queue()
    for current_epoch in range(start_epoch, p.num_epochs, 1):
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
        try:
            if current_epoch % 2 == 0:
                head_id = 0
            else:
                head_id = 1 + (current_epoch - math.ceil(current_epoch / 2)) % (len(heads) - 1)
            head_id = 0
            head_name = p.species[0]
            # check_perf()
            # exit()
            # load_old_weights()
            # exit()
            if make_data_proc is None:
                make_data_proc = mp.Process(target=get_train_data, args=(mp_q, head_id,))
                make_data_proc.start()

            saved_train_data = mp_q.get(timeout=1000)

            train_data = mo.CustomDataset(saved_train_data[0], saved_train_data[1])
            train_dataloader = DataLoader(train_data, batch_size=p.GLOBAL_BATCH_SIZE)

            make_data_proc = mp.Process(target=get_train_data, args=(mp_q, head_id,))
            make_data_proc.start()
            train(train_dataloader, p.species[head_id])
            if current_epoch % 5 == 0 and current_epoch != 0:
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, p.model_folder + p.model_name)
            if current_epoch % 50 == 0 and current_epoch != 0:  # and current_epoch != 0:
                print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating.")
                check_perf()
                with open(f"dry_test.bed", "a+") as text_file:
                    text_file.write("\n".join(dry_run_regions))
        except Exception as e:
            print(f"Problem with the epoch {current_epoch}!")
            traceback.print_exc()
            make_data_proc = None

