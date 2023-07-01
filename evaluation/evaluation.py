import joblib
import gc
import numpy as np
from scipy import stats
import math
import common as cm
import parse_data as parser
import pandas as pd
import torch
from torch.utils.data import DataLoader
import model as mo


def eval_perf(p, model, device, head, eval_infos_all, should_draw, current_epoch, label, one_hot):
    print("Version 1.02")
    eval_track_names = []
    if isinstance(head, dict):
        for key in head.keys():
            eval_track_names += head[key]
    else:
        eval_track_names = head
    eval_infos = []
    for info in eval_infos_all:
        if info[5]:
            continue
        eval_infos.append(info)

    load_info = []
    for j, info in enumerate(eval_infos):
        mid = int(info[1] / p.bin_size)
        load_info.append([info[0], mid])
    print("Loading ground truth tracks")
    gt = parser.par_load_data(load_info, eval_track_names, p)

    print("Extracting DNA regions")
    test_seq = []
    for info in eval_infos:
        start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
        extra = start + p.input_size - len(one_hot[info[0]])
        if start < 0:
            ns = one_hot[info[0]][0:start + p.input_size]
            ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
        elif extra > 0:
            ns = one_hot[info[0]][start: len(one_hot[info[0]])]
            ns = np.concatenate((ns, np.zeros((extra, 5))))
        else:
            ns = one_hot[info[0]][start:start + p.input_size]
        test_seq.append(ns[:, :-1])
    test_seq = np.asarray(test_seq, dtype=bool)
    print(f"Length: {len(test_seq)}")
    meta = pd.read_csv("data/all_track.metadata.tsv", sep="\t")
    track_types = {}
    for track in eval_track_names:
        meta_row = meta.loc[meta['file_name'] == track]
        if len(meta_row) > 0:
            track_types[track] = meta_row.iloc[0]["technology"]
        else:
            track_types[track] = "scEnd5"
    print("Predicting")
    model.eval()
    # predictions = joblib.load("pred.gz")
    dd = mo.DatasetDNA(test_seq)
    ddl = DataLoader(dataset=dd, batch_size=p.pred_batch_size, shuffle=False)
    for batch, X in enumerate(ddl):
        print(batch, end=" ")
        with torch.no_grad():
            pr = model(X)
        p1 = np.concatenate((pr['hg38_expression'].cpu().numpy(),
                             pr['hg38_epigenome'].cpu().numpy(),
                             pr['hg38_conservation'].cpu().numpy()), axis=2)
        p2 = p1[:, p.mid_bin, :]  # p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
        if batch == 0:
            predictions = p2
        else:
            predictions = np.concatenate((predictions, p2), dtype=np.float16)
    # joblib.dump(predictions, "pred.gz", compress="lz4")
    u_track_types = set(track_types.values())

    def eval_perf(expressed_only=False):
        # Across tracks
        at_corrs_p = {}
        at_corrs_s = {}
        for track_type in u_track_types:
            for i in range(len(eval_infos)):
                a = []
                b = []
                for j, track in enumerate(eval_track_names):
                    if track_types[track] != track_type:
                        continue
                    if expressed_only and gt[i, j] == 0:
                        continue
                    a.append(predictions[i, j])
                    b.append(gt[i, j])
                if len(b) < 5:
                    continue
                a = np.nan_to_num(a, neginf=0, posinf=0)
                b = np.nan_to_num(b, neginf=0, posinf=0)
                pc = stats.pearsonr(a, b)[0]
                sc = stats.spearmanr(a, b)[0]
                if not math.isnan(sc) and not math.isnan(pc):
                    at_corrs_p.setdefault(track_type, []).append(pc)
                    at_corrs_s.setdefault(track_type, []).append(sc)

        # Across genes
        corrs_p = {}
        corrs_s = {}
        all_track_spearman = {}
        for i, track in enumerate(eval_track_names):
            a = []
            b = []
            for j in range(len(eval_infos)):
                if expressed_only and gt[j, i] == 0:
                    continue
                a.append(predictions[j, i])
                b.append(gt[j, i])
            if len(b) < 5:
                continue
            a = np.nan_to_num(a, neginf=0, posinf=0)
            b = np.nan_to_num(b, neginf=0, posinf=0)
            pc = stats.pearsonr(a, b)[0]
            sc = stats.spearmanr(a, b)[0]
            if not math.isnan(sc) and not math.isnan(pc):
                corrs_p.setdefault(track_types[track], []).append((pc, track))
                corrs_s.setdefault(track_types[track], []).append((sc, track))
                all_track_spearman[track] = stats.spearmanr(a, b)[0]

        eval_log = ""
        if expressed_only:
            eval_log += "Expressed only\n"
        else:
            eval_log += "All values\n"
        eval_log += "Type\tCount\tAcross genes PCC\tAcross genes SC\tAcross tracks PCC\tAcross tracks SC\n"
        for track_type in corrs_p.keys():
            with open(f"all_track_spearman_{track_type}_tss.csv", "w+") as myfile:
                for ccc in corrs_s[track_type]:
                    myfile.write(f"{ccc[0]},{ccc[1]}")
                    myfile.write("\n")
            type_pcc = [i[0] for i in corrs_p[track_type]]
            eval_log += f"{track_type}\t{len(type_pcc)}\t{np.mean(type_pcc):.2f}\t{np.mean([i[0] for i in corrs_s[track_type]]):.2f}\t{np.mean(at_corrs_p[track_type]):.2f}\t{np.mean(at_corrs_s[track_type]):.2f}\n"

        if not expressed_only:
            with open("all_track_spearman_tss.csv", "w+") as myfile:
                for key in all_track_spearman.keys():
                    myfile.write(f"{key},{all_track_spearman[key]}")
                    myfile.write("\n")
        return eval_log, f"{np.mean([i[0] for i in corrs_s['CAGE']])}_{np.mean(at_corrs_s['CAGE'])}"

    # l1, _ = eval_perf(True)
    l2, return_result = eval_perf(False)
    # log = l1 + "\n\n" + l2
    log = l2
    print(log)
    with open("log.txt", "a") as myfile:
        myfile.write(log + "\n")
    # if should_draw:
    #     viz.draw_regplots(cor_tracks, all_track_spearman, final_pred, eval_gt,
    #                       f"{p.figures_folder}/plots/epoch_{current_epoch}")

    return return_result


def change_seq(x):
    return cm.rev_comp(x)