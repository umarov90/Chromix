import joblib
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from scipy import stats
import math
import visualization as viz
import common as cm
import parse_data as parser
import pandas as pd
from collections import defaultdict


def eval_perf(p, our_model, head, eval_infos_all, should_draw, current_epoch, label, one_hot):
    import model as mo
    print("Version 1.02")
    eval_track_names = []
    for key in head.keys():
        eval_track_names += head[key]
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
    print("Extracting evaluation regions")
    eval_gt = {}
    for i in range(len(eval_infos)):
        eval_gt[eval_infos[i][2]] = {}

    tss_count = {}
    tss_count = defaultdict(lambda:0, tss_count)
    for i, info in enumerate(eval_infos):
        tss_count[info[2]] = tss_count[info[2]] + 1
        for j, track in enumerate(eval_track_names):
            eval_gt[info[2]].setdefault(track, []).append(gt[i, j])

    for i, gene in enumerate(eval_gt.keys()):
        if i % 10 == 0:
            print(i, end=" ")
        for track in eval_track_names:
            eval_gt[gene][track] = np.sum(eval_gt[gene][track])
    print("")

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

    start_val = {}
    track_inds_bed = []
    meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
    cor_tracks = pd.read_csv("data/fantom_tracks.tsv", sep="\t", header=None).iloc[:, 0].tolist()
    print(f"Number of correlation tracks: {len(cor_tracks)}")
    track_types = {}
    for track in eval_track_names:
        meta_row = meta.loc[meta['file_name'] == track]
        if len(meta_row) > 0:
            track_types[track] = meta_row.iloc[0]["technology"]
        else:
            track_types[track] = "scEnd5"
    print(f"Number of tracks for bed: {len(track_inds_bed)}")
    print("Predicting")
    # predictions = joblib.load("pred.gz")
    for w in range(0, len(test_seq), p.w_step):
        print(w, end=" ")
        pr = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
        if len(head.keys()) > 1:
            p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        else:
            p1 = pr
        p2 = p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
        if w == 0:
            predictions = p2
        else:
            predictions = np.concatenate((predictions, p2), dtype=np.float16)
        predictions_for_bed = p1
    
        p1 = None
        p2 = None
        predictions_for_bed = None
        gc.collect()
    # joblib.dump(predictions, "pred.gz", compress="lz4")
    final_pred = {}
    for i in range(len(eval_infos)):
        final_pred[eval_infos[i][2]] = {}
    for i in range(len(eval_infos)):
        for it, track in enumerate(eval_track_names):
            final_pred[eval_infos[i][2]].setdefault(track, []).append(predictions[i][it])

    for i, gene in enumerate(final_pred.keys()):
        if i % 10 == 0:
            print(i, end=" ")
        for track in eval_track_names:
            final_pred[gene][track] = np.sum(final_pred[gene][track])

    # Across tracks
    at_corrs_p = {}
    at_corrs_s = {}
    genes_performance = []
    u_track_types = set(track_types.values())
    for track_type in u_track_types:
        for gene in final_pred.keys():
            a = []
            b = []
            indices = []
            for v, track in enumerate(eval_track_names):
                if track_types[track] != track_type:
                    continue
                a.append(final_pred[gene][track])
                b.append(eval_gt[gene][track])
                if track_types[track] == "CAGE" and "FANTOM5" in track:
                    indices.append(v)
            a = np.nan_to_num(a, neginf=0, posinf=0)
            b = np.nan_to_num(b, neginf=0, posinf=0)
            pc = stats.pearsonr(a, b)[0]
            sc = stats.spearmanr(a, b)[0]
            if not math.isnan(sc) and not math.isnan(pc):
                at_corrs_p.setdefault(track_type, []).append(pc)
                at_corrs_s.setdefault(track_type, []).append(sc)
                if track_types[track] == "CAGE" and "FANTOM5" in track:
                    df = pd.DataFrame({'Prediction': a, 'GT': b})
                    df.to_csv(f"genes/{gene}.tsv", index=False, sep="\t")
                    genes_performance.append(f"{gene}\t{sc}\t{np.mean(b)}\t{np.std(b)}\t{tss_count[gene]}\t{eval_track_names[indices[np.argmin(a)]]}\t{eval_track_names[indices[np.argmax(a)]]}")

    with open(f"genes_performance_{label}.tsv", 'w+') as f:
        f.write('\n'.join(genes_performance))

    # Across genes
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}
    for track in eval_track_names:
        a = []
        b = []
        for gene in eval_gt.keys():
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)            
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        track_perf[track] = pc
        if not math.isnan(sc) and not math.isnan(pc):
            corrs_p.setdefault(track_types[track], []).append((pc, track))
            corrs_s.setdefault(track_types[track], []).append((sc, track))
            all_track_spearman[track] = stats.spearmanr(a, b)[0]

    with open("all_track_spearman_tss.csv", "w+") as myfile:
        for key in all_track_spearman.keys():
            myfile.write(f"{key},{all_track_spearman[key]}")
            myfile.write("\n")

    print("Type\tCount\tAcross genes PCC\tAcross genes SC\tAcross tracks PCC\tAcross tracks SC")
    for track_type in corrs_p.keys():
        with open(f"all_track_spearman_{track_type}_tss.csv", "w+") as myfile:
            for ccc in corrs_s[track_type]:
                myfile.write(f"{ccc[0]},{ccc[1]}")
                myfile.write("\n")
        type_pcc = [i[0] for i in corrs_p[track_type]]
        print(f"{track_type}\t{len(type_pcc)}\t{np.mean(type_pcc):.2f}\t"
              f"{np.mean([i[0] for i in corrs_s[track_type]]):.2f}\t"
              f"{np.mean(at_corrs_p[track_type]):.2f}\t{np.mean(at_corrs_s[track_type]):.2f}")

    return_result = f"{np.mean([i[0] for i in corrs_s['CAGE']])}_{np.mean(at_corrs_s['CAGE'])}"

    if should_draw:
        viz.draw_regplots(cor_tracks, track_perf, final_pred, eval_gt,
                          f"{p.figures_folder}/plots/epoch_{current_epoch}")

    return return_result


def change_seq(x):
    return cm.rev_comp(x)