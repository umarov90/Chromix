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


def eval_perf(p, our_model, head, eval_infos, should_draw, current_epoch, label, one_hot):
    import model as mo
    eval_track_names = []
    for key in head.keys():
        eval_track_names += head[key]

    if Path(f"{p.pickle_folder}{label}_seq.gz").is_file():
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading sequences. ")
        test_seq = joblib.load(f"{p.pickle_folder}/{label}_seq.gz")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 1. ")
        # eval_gt = joblib.load(f"{p.pickle_folder}/{chr_name}_eval_gt.gz")
        eval_gt = pickle.load(open(f"{p.pickle_folder}/{label}_eval_gt.gz", "rb"))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 2. ")
        # eval_gt_tss = joblib.load(f"{p.pickle_folder}/{label}_eval_gt_tss.gz")
        eval_gt_tss = pickle.load(open(f"{p.pickle_folder}/{label}_eval_gt_tss.gz", "rb"))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Finished loading. ")
    else:
        load_info = []
        for j, info in enumerate(eval_infos):
            mid = int(info[1] / p.bin_size)
            load_info.append([info[0], mid])
        print("Loading ground truth tracks")
        gt = parser.par_load_data(load_info, eval_track_names, p)
        print("Extracting evaluation regions")
        eval_gt = {}
        eval_gt_tss = {}
        for i in range(len(eval_infos)):
            eval_gt[eval_infos[i][2]] = {}

        for i, info in enumerate(eval_infos):
            for j, track in enumerate(eval_track_names):
                eval_gt_tss.setdefault(track, []).append(gt[i, j])
                eval_gt[info[2]].setdefault(track, []).append(gt[i, j])

        for track in eval_track_names:
            for gene in eval_gt.keys():
                eval_gt[gene][track] = np.mean(eval_gt[gene][track])
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
        print(f"Lengths: {len(test_seq)} {len(eval_gt)}")
        gc.collect()
        print("Dumping the evaluation data")
        joblib.dump(test_seq, f"{p.pickle_folder}/{label}_seq.gz", compress="lz4")
        # pickle.dump(test_seq, open(f"{p.pickle_folder}/{chr_name}_seq.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()
        # joblib.dump(eval_gt, f"{p.pickle_folder}/{chr_name}_eval_gt.gz", compress="lz4")
        pickle.dump(eval_gt, open(f"{p.pickle_folder}/{label}_eval_gt.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()
        # joblib.dump(eval_gt_tss, f"{p.pickle_folder}/{label}_eval_gt_tss.gz", compress="lz4")
        pickle.dump(eval_gt_tss, open(f"{p.pickle_folder}/{label}_eval_gt_tss.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    final_pred = {}
    for i in range(len(eval_infos)):
        final_pred[eval_infos[i][2]] = {}

    for w in range(0, len(test_seq), p.w_step):
        print(w, end=" ")
        pr = our_model.predict(mo.wrap2(test_seq[w:w + p.w_step], p.predict_batch_size))
        p1 = np.concatenate((pr[0], pr[1], pr[2]), axis=1)
        p2 = p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
        if w == 0:
            predictions = p2
        else:
            predictions = np.concatenate((predictions, p2), dtype=np.float32)
        gc.collect()

    protein_gene_set = []
    final_pred_tss = {}
    for i in range(len(eval_infos)):
        if not eval_infos[i][5]:
            protein_gene_set.append(eval_infos[i][2])
        for it, track in enumerate(eval_track_names):
            # Grouping TSS into genes
            final_pred[eval_infos[i][2]].setdefault(track, []).append(predictions[i][it])
            final_pred_tss.setdefault(track, []).append(predictions[i][it])
    for i, gene in enumerate(final_pred.keys()):
        if i % 10 == 0:
            print(i, end=" ")
        for track in eval_track_names:
            final_pred[gene][track] = np.mean(final_pred[gene][track])

    corr_p = []
    corr_s = []
    genes_performance = []
    for gene in final_pred.keys():
        a = []
        b = []
        for track in eval_track_names:
            type = track[:track.find(".")]
            if type != "CAGE":
                continue
            # if track not in eval_tracks:
            #     continue
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        if not math.isnan(sc) and not math.isnan(pc):
            corr_p.append(pc)
            corr_s.append(sc)
            genes_performance.append(f"{gene}\t{sc}\t{np.mean(b)}\t{np.std(b)}")

    print("")
    print(f"Across tracks {len(corr_p)} {np.mean(corr_p)} {np.mean(corr_s)}")
    with open("genes_performance.tsv", 'w+') as f:
        f.write('\n'.join(genes_performance))

    print("Across Genes")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}

    for track in eval_track_names:
        type = track[:track.find(".")]
        a = []
        b = []
        for gene in protein_gene_set:
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        # if "hela" in track.lower():
        #     print(f"{track}\t{sc}")
        track_perf[track] = sc
        if pc is not None and sc is not None:
            corrs_p.setdefault(type, []).append((pc, track))
            corrs_s.setdefault(type, []).append((sc, track))
            all_track_spearman[track] = stats.spearmanr(a, b)[0]

    with open("all_track_spearman.csv", "w+") as myfile:
        for key in all_track_spearman.keys():
            myfile.write(f"{key},{all_track_spearman[key]}")
            myfile.write("\n")

    for track_type in corrs_p.keys():
        with open(f"all_track_spearman_{track_type}.csv", "w+") as myfile:
            for ccc in corrs_s[track_type]:
                myfile.write(f"{ccc[0]},{ccc[1]}")
                myfile.write("\n")
        type_pcc = [i[0] for i in corrs_p[track_type]]
        print(f"{track_type} correlation : {np.mean(type_pcc)}"
              f" {np.mean([i[0] for i in corrs_s[track_type]])} {len(type_pcc)}")

    with open("result_cage_test.csv", "w+") as myfile:
        for ccc in corrs_p["CAGE"]:
            myfile.write(str(ccc[0]) + "," + str(ccc[1]))
            myfile.write("\n")

    with open("result.txt", "a+") as myfile:
        myfile.write(datetime.now().strftime('[%H:%M:%S] ') + "\n")
        for track_type in corrs_p.keys():
            myfile.write(str(track_type) + "\t")
        for track_type in corrs_p.keys():
            myfile.write(str(np.mean([i[0] for i in corrs_p[track_type]])) + "\t")
        myfile.write("\n")

    print("Across genes (TSS)")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}
    print(f"Number of TSS: {len(eval_gt_tss[eval_track_names[0]])}")
    for track in eval_track_names:
        type = track[:track.find(".")]
        a = eval_gt_tss[track]
        b = final_pred_tss[track]
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        track_perf[track] = pc
        if pc is not None and sc is not None:
            corrs_p.setdefault(type, []).append((pc, track))
            corrs_s.setdefault(type, []).append((sc, track))
            all_track_spearman[track] = stats.spearmanr(a, b)[0]

    with open("all_track_spearman_tss.csv", "w+") as myfile:
        for key in all_track_spearman.keys():
            myfile.write(f"{key},{all_track_spearman[key]}")
            myfile.write("\n")

    for track_type in corrs_p.keys():
        with open(f"all_track_spearman_{track_type}_tss.csv", "w+") as myfile:
            for ccc in corrs_s[track_type]:
                myfile.write(f"{ccc[0]},{ccc[1]}")
                myfile.write("\n")
        type_pcc = [i[0] for i in corrs_p[track_type]]
        print(f"{track_type} correlation : {np.mean(type_pcc)}"
              f" {np.mean([i[0] for i in corrs_s[track_type]])} {len(type_pcc)}")
    return_result = np.mean([i[0] for i in corrs_s["CAGE"]])

    if should_draw:
        viz.draw_regplots(eval_track_names, track_perf, final_pred, eval_gt,
                          f"{p.figures_folder}/plots/epoch_{current_epoch}")

    return return_result


def change_seq(x):
    return cm.rev_comp(x)
