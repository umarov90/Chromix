import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
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
from torch import autocast, nn, optim
from sync_batchnorm import convert_model

matplotlib.use("agg")


def get_train_data(q, num_seq, tss_only):
    train_data = {}
    for specie in p.species:
        # training regions are shuffled each iteration
        if specie == "hg38" and tss_only:
            shuffled_regions_info = train_info
            shuffled_regions_info = random.sample(shuffled_regions_info, len(shuffled_regions_info))
        else:
            shuffled_regions_info = random.sample(training_regions[specie], len(training_regions[specie]))
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

        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
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

        if specie == "hg38":
            output_conservation = parser.par_load_data(output_scores_info, heads[specie]["conservation"], p)
        output_expression = parser.par_load_data(output_scores_info, heads[specie]["expression"], p)
        # output_expression_sc = parser.par_load_data(output_scores_info, heads[specie]["expression_sc"], p)
        output_epigenome = parser.par_load_data(output_scores_info, heads[specie]["epigenome"], p)
        # loading corresponding 2D data
        output_hic = parser.par_load_hic_data(p.hic_keys[specie], p, picked_regions, half)
        # for reverse-complement sequences, the 1D output is flipped
        for i in range(half):
            output_expression[i] = np.flip(output_expression[i], axis=1)
            # output_expression_sc[i] = np.flip(output_expression_sc[i], axis=1)
            output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
            if specie == "hg38":
                output_conservation[i] = np.flip(output_conservation[i], axis=1)

        all_outputs = {"expression": output_expression, "epigenome": output_epigenome, "hic": output_hic} # "expression_sc": output_expression_sc
        if specie == "hg38":
            all_outputs["conservation"] = output_conservation
        train_data[specie] = (input_sequences, all_outputs)
    q.put(train_data)


def train(dataloader):
    print("--------------------------------------------------------")
    # p.loss_weights["hic"] = min(p.loss_weights["hic"] * 1.01, 8.0)
     #print(f"hic weight: {p.loss_weights['hic']}")
    if not p.tss_only:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = 1e-04
    elif p.tss_only:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = 1e-05
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        total_loss = 0
        for specie in p.species:
            if eval_epoch and specie != "hg38":
                continue
            ydd = {k: v for k, v in y.items() if k.startswith(specie)}
            loss = model(X[specie], target=ydd, head=specie, loss_weights=p.loss_weights).mean()
            if not torch.isnan(loss):
                loss.backward()
                total_loss += loss.item()
            else:
                del loss
                print("Skipping because of nan!")
                break
            # torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
            mo.individual_clip(model.module.stem.parameters())
            mo.individual_clip(model.module.conv_tower.parameters())
            mo.individual_clip(model.module.convnext.parameters())
            mo.individual_clip(model.module.final_pointwise.parameters())
            mo.individual_clip(model.module.hic_projection.parameters())

            optimizer.step()
            optimizer.zero_grad()
        p.running_loss["total"].append(total_loss)
        print(f"Total loss: {np.median(p.running_loss['total'][-1000:]):.6f}")
        for specie in p.species:
            if eval_epoch and specie != "hg38":
                continue
            print(specie, end=" loss ")
            for k in p.running_loss[specie].keys():
                print(f"{k}: {np.median(p.running_loss[specie][k][-2000:]):.6f}", end=" ")
            print()
        print(f"{batch + 1}/{len(dataloader)} =====================================================================")


def check_perf():
    try:
        auc = 0  # get_linking_AUC()
        train_eval_info = random.sample(train_info, len(train_info) // 10)
        print(f"Training set {len(train_eval_info)}")
        training_result = evaluation.eval_perf(p, model, device, heads["hg38"], train_eval_info,
                                               False, current_epoch, "train", one_hot["hg38"])
        # training_result = "0"
        valid_eval_chr_info = []
        for info in valid_info:
            # if info[0] == "chr2":
            valid_eval_chr_info.append(info)
        print(f"Valid set {len(valid_eval_chr_info)}")
        valid_result = evaluation.eval_perf(p, model, device, heads["hg38"], valid_eval_chr_info,
                                            False, current_epoch, "valid", one_hot["hg38"])
        with open(p.model_name + "_history.tsv", "a+") as myfile:
            myfile.write(training_result + "\t" + valid_result + "\t" + str(auc) + "\n")
        new_folder = p.model_folder + valid_result + "_" + str(auc) + "/"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, new_folder + p.model_name)
        target_dir = os.path.join(script_dir, valid_result)
        os.makedirs(target_dir)
        for py_file in py_files.keys():
            with open(os.path.join(target_dir, py_file), 'w') as file:
                file.write(py_files[py_file])
    except Exception as e:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")


make_data_proc = None
script_dir = os.path.dirname(os.path.realpath(__file__))
py_files = {}
for py_file in ['model.py', 'main.py', 'parse_data.py', 'main_params.py']:
    with open(os.path.join(script_dir, py_file), 'r') as file:
        py_files[py_file] = file.read()
p = MainParams()
dry_run_regions = []
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    train_info, valid_info, test_info = parser.parse_sequences(p)
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
                                               ("CAGE", "NETCAGE"))],
                            "epigenome": [x for x in shuffled_tracks if
                                          meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                              ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))]
                            }
                if specie == "hg38":
                    new_head["conservation"] = [x for x in shuffled_tracks if
                                                meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
                                                    "conservation")]
            else:
                new_head = shuffled_tracks
            heads[specie] = new_head
        joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress=3)
    for head_key in heads.keys():
        if isinstance(heads[head_key], dict):
            for key2 in heads[head_key].keys():
                print(f"Number of tracks in head {head_key} {key2}: {len(heads[head_key][key2])}")
                p.output_heads[head_key + "_" + key2] = len(heads[head_key][key2])
        else:
            print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")
            p.output_heads[head_key + "_expression"] = len(heads[head_key])

    training_regions = {}
    one_hot = {}
    for specie in p.species:
        training_regions[specie] = joblib.load(f"{p.pickle_folder}{specie}_regions.gz")
        one_hot[specie] = joblib.load(f"{p.pickle_folder}{specie}_one_hot.gz")

    print("Training starting")
    model, optimizer = mo.prepare_model(p)
    start_epoch = mo.load_weights(p, model, optimizer)
    # mo.reset_batchnorm_stats(model)
    print(model.module)
    fit_epochs = 1
    aux_species = p.species[2:]
    mp_q = mp.Queue()
    for current_epoch in range(start_epoch, p.num_epochs, 1):
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch))
        try:
            eval_epoch = (current_epoch + 1) % 1000000 == 0
            check_perf()
            exit()
            if make_data_proc is None:
                num_seq = p.BATCH_SIZE * (p.MAX_STEPS_PER_EPOCH - p.MAX_STEPS_PER_EPOCH % p.accum_iter)
                make_data_proc = mp.Process(target=get_train_data, args=(mp_q, num_seq, p.tss_only,))
                make_data_proc.start()

            saved_train_data = mp_q.get(timeout=10000)
            train_data = mo.CustomDataset(saved_train_data)
            train_dataloader = DataLoader(train_data, batch_size=p.BATCH_SIZE)
            cm.print_memory()

            num_seq = p.BATCH_SIZE * (p.MAX_STEPS_PER_EPOCH - p.MAX_STEPS_PER_EPOCH % p.accum_iter)
            make_data_proc = mp.Process(target=get_train_data, args=(mp_q, num_seq, p.tss_only,))
            make_data_proc.start()

            train(train_dataloader)
            if not eval_epoch:
                torch.save({
                    'epoch': current_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, p.model_folder + p.model_name)
            if eval_epoch:
                print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
                check_perf()
                with open(f"dry_test.bed", "a+") as text_file:
                    text_file.write("\n".join(dry_run_regions))
                exit()
        except Exception as e:
            print(f"Problem with the epoch {current_epoch}!")
            traceback.print_exc()
            make_data_proc = None
