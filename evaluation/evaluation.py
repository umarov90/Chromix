import joblib
import gc
import numpy as np
from scipy import stats
import math
import common as cm
import parse_data as parser
import pandas as pd


def eval_perf(p, our_model, head, eval_infos_all, should_draw, current_epoch, label, one_hot):
    import model as mo
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
    meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
    track_types = {}
    for track in eval_track_names:
        meta_row = meta.loc[meta['file_name'] == track]
        if len(meta_row) > 0:
            track_types[track] = meta_row.iloc[0]["technology"]
        else:
            track_types[track] = "scEnd5"
    print("Predicting")
    # predictions = joblib.load("pred.gz")
    for w in range(0, len(test_seq), p.w_step):
        print(w, end=" ")
        pr = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
        if isinstance(head, dict):
            p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        else:
            p1 = pr
        p2 = p1[:, :, p.mid_bin] # p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
        if w == 0:
            predictions = p2
        else:
            predictions = np.concatenate((predictions, p2), dtype=np.float16)
        p1 = None
        p2 = None
        gc.collect()
    # joblib.dump(predictions, "pred.gz", compress="lz4")

    # Across tracks
    at_corrs_p = {}
    at_corrs_s = {}
    u_track_types = set(track_types.values())
    for track_type in u_track_types:
        for i in range(len(eval_infos)):
            a = []
            b = []
            indices = []
            for j, track in enumerate(eval_track_names):
                if track_types[track] != track_type:
                    continue
                a.append(predictions[i,j])
                b.append(gt[i,j])
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
        # if "FANTOM5" in track:
        #     print(track)
        #     continue
        a = []
        b = []
        for j in range(len(eval_infos)):
            a.append(predictions[j,i])
            b.append(gt[j,i])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)            
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
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

    # if should_draw:
    #     viz.draw_regplots(cor_tracks, all_track_spearman, final_pred, eval_gt,
    #                       f"{p.figures_folder}/plots/epoch_{current_epoch}")

    return return_result


def change_seq(x):
    return cm.rev_comp(x)