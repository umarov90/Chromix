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


def eval_perf(p, our_model, eval_track_names, eval_infos, should_draw, current_epoch, label, one_hot,
              loaded_tracks):
    import model as mo
    print("Model loaded")
    predict_batch_size = p.GLOBAL_BATCH_SIZE
    w_step = 500

    if Path(f"pickle/{label}_seq.gz").is_file():
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading sequences. ")
        test_seq = joblib.load(f"pickle/{label}_seq.gz")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 1. ")
        # eval_gt = joblib.load(f"pickle/{chr_name}_eval_gt.gz")
        eval_gt = pickle.load(open(f"pickle/{label}_eval_gt.gz", "rb"))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 2. ")
        eval_gt_tss = joblib.load(f"pickle/{label}_eval_gt_tss.gz")
        # eval_gt_tss = pickle.load(open(f"pickle/{chr_name}_eval_gt_tss.gz", "rb"))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Finished loading. ")
    else:
        eval_gt = {}
        eval_gt_tss = {}
        for i in range(len(eval_infos)):
            eval_gt[eval_infos[i][2]] = {}
        for i, key in enumerate(eval_track_names):
            if i % 100 == 0:
                print(i, end=" ")
                gc.collect()
            if key in loaded_tracks.keys():
                parsed_track = loaded_tracks[key]
            else:
                parsed_track = joblib.load(p.parsed_tracks_folder + key)
                # with open(p.parsed_tracks_folder + key, 'rb') as fp:
                #     parsed_track = pickle.load(fp)
            mids = []
            for j, info in enumerate(eval_infos):
                mid = int(info[1] / p.bin_size)
                # if mid in mids:
                #     continue
                # mids.append(mid)
                # val = parsed_track[info[0]][mid]
                val = parsed_track[info[0]][mid - 1] + parsed_track[info[0]][mid] + \
                      parsed_track[info[0]][mid + 1] + parsed_track[info[0]][mid + 2] + parsed_track[info[0]][mid - 2]
                eval_gt_tss.setdefault(key, []).append(val)
                eval_gt[info[2]].setdefault(key, []).append(val)
            if i == 0:
                print(f"Skipped: {len(eval_infos) - len(mids)}")

        for i, gene in enumerate(eval_gt.keys()):
            if i % 10 == 0:
                print(i, end=" ")
            for track in eval_track_names:
                eval_gt[gene][track] = np.mean(eval_gt[gene][track])
        print("")

        for key in eval_gt_tss.keys():
            eval_gt_tss[key] = np.asarray(eval_gt_tss[key], dtype=np.float16)

        test_seq = []
        starts = []
        for info in eval_infos:
            start = int(info[1] - (info[1] % p.bin_size) - p.half_size)
            # if start in starts:
            #     continue
            # starts.append(start)
            extra = start + p.input_size - len(one_hot[info[0]])
            if start < 0:
                ns = one_hot[info[0]][0:start + p.input_size]
                ns = np.concatenate((np.zeros((-1 * start, 5)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, 5))))
            else:
                ns = one_hot[info[0]][start:start + p.input_size]
            if len(ns) != p.input_size:
                print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
            test_seq.append(ns[:, :-1])
        print(f"Skipped: {len(eval_infos) - len(starts)}")
        test_seq = np.asarray(test_seq, dtype=bool)
        print(f"Lengths: {len(test_seq)} {len(eval_gt)}")
        gc.collect()
        joblib.dump(test_seq, f"pickle/{label}_seq.gz", compress="lz4")
        # pickle.dump(test_seq, open(f"pickle/{chr_name}_seq.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del test_seq
        gc.collect()
        # joblib.dump(eval_gt, f"pickle/{chr_name}_eval_gt.gz", compress="lz4")
        pickle.dump(eval_gt, open(f"pickle/{label}_eval_gt.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del eval_gt
        gc.collect()
        joblib.dump(eval_gt_tss, f"pickle/{label}_eval_gt_tss.gz", compress="lz4")
        # pickle.dump(eval_gt_tss, open(f"pickle/{chr_name}_eval_gt_tss.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        test_seq = joblib.load(f"pickle/{label}_seq.gz")
        eval_gt = joblib.load(f"pickle/{label}_eval_gt.gz")

    final_pred = {}
    for i in range(len(eval_infos)):
        final_pred[eval_infos[i][2]] = {}

    for w in range(0, len(test_seq), w_step):
        print(w, end=" ")
        p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size), batch_size=predict_batch_size)
        p2 = p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1] + p1[:, :, p.mid_bin + 2] + p1[:, :, p.mid_bin - 2]
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
    print(f"Across genes {len(corr_p)} {np.mean(corr_p)} {np.mean(corr_s)}")
    with open("genes_performance.tsv", 'w+') as f:
        f.write('\n'.join(genes_performance))

    print("Across tracks (Genes)")
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

    print("Across tracks (TSS)")
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
