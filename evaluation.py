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


def eval_perf(p, our_model, eval_track_names, eval_infos, should_draw, current_epoch, chr_name, one_hot, hic_keys, loaded_tracks):
    import model as mo
    print("Model loaded")
    predict_batch_size = p.GLOBAL_BATCH_SIZE
    w_step = 500
    full_preds_steps_num = 1

    predictions_hic = []

    if Path(f"pickle/{chr_name}_seq.gz").is_file():
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading sequences. ")
        test_seq = joblib.load(f"pickle/{chr_name}_seq.gz")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 1. ")
        # eval_gt = joblib.load(f"pickle/{chr_name}_eval_gt.gz")
        eval_gt = pickle.load(open(f"pickle/{chr_name}_eval_gt.gz", "rb"))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 2. ")
        eval_gt_tss = joblib.load(f"pickle/{chr_name}_eval_gt_tss.gz")
        # eval_gt_tss = pickle.load(open(f"pickle/{chr_name}_eval_gt_tss.gz", "rb"))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loading gt 3. ")
        eval_gt_full = joblib.load(f"pickle/{chr_name}_eval_gt_full.gz")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Finished loading. ")
    else:
        eval_gt = {}
        eval_gt_full = []
        eval_gt_tss = {}
        for i in range(len(eval_infos)):
            eval_gt[eval_infos[i][2]] = {}
            if i < full_preds_steps_num * w_step:
                eval_gt_full.append([])
        for i, key in enumerate(eval_track_names):
            if i % 100 == 0:
                print(i, end=" ")
                gc.collect()
            if key in loaded_tracks.keys():
                parsed_track = loaded_tracks[key]
            else:
                # parsed_track = joblib.load(parsed_tracks_folder + key)
                with open(p.parsed_tracks_folder + key, 'rb') as fp:
                    parsed_track = pickle.load(fp)
            mids = []
            for j, info in enumerate(eval_infos):
                mid = int(info[1] / p.bin_size)
                # if mid in mids:
                #     continue
                # mids.append(mid)
                # val = parsed_track[info[0]][mid]
                val = parsed_track[info[0]][mid - 1] + parsed_track[info[0]][mid] + parsed_track[info[0]][mid + 1]
                eval_gt_tss.setdefault(key, []).append(val)
                eval_gt[info[2]].setdefault(key, []).append(val)
                if j < full_preds_steps_num * w_step:
                    start_bin = int(info[1] / p.bin_size) - p.half_num_regions
                    extra_bin = start_bin + p.num_regions - len(parsed_track[info[0]])
                    if start_bin < 0:
                        binned_region = parsed_track[info[0]][0: start_bin + p.num_regions]
                        binned_region = np.concatenate((np.zeros(-1 * start_bin), binned_region))
                    elif extra_bin > 0:
                        binned_region = parsed_track[info[0]][start_bin: len(parsed_track[info[0]])]
                        binned_region = np.concatenate((binned_region, np.zeros(extra_bin)))
                    else:
                        binned_region = parsed_track[info[0]][start_bin:start_bin + p.num_regions]
                    eval_gt_full[j].append(binned_region)
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

        eval_gt_full = np.asarray(eval_gt_full)

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
                ns = np.concatenate((np.zeros((-1 * start, p.num_features)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, p.num_features))))
            else:
                ns = one_hot[info[0]][start:start + p.input_size]
            if len(ns) != p.input_size:
                print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
            test_seq.append(ns)
        print(f"Skipped: {len(eval_infos) - len(starts)}")
        test_seq = np.asarray(test_seq, dtype=bool)
        print(f"Lengths: {len(test_seq)} {len(eval_gt)} {len(eval_gt_full)}")
        gc.collect()
        joblib.dump(test_seq, f"pickle/{chr_name}_seq.gz", compress="lz4")
        # pickle.dump(test_seq, open(f"pickle/{chr_name}_seq.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del test_seq
        gc.collect()
        # joblib.dump(eval_gt, f"pickle/{chr_name}_eval_gt.gz", compress="lz4")
        pickle.dump(eval_gt, open(f"pickle/{chr_name}_eval_gt.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del eval_gt
        gc.collect()
        joblib.dump(eval_gt_tss, f"pickle/{chr_name}_eval_gt_tss.gz", compress="lz4")
        # pickle.dump(eval_gt_tss, open(f"pickle/{chr_name}_eval_gt_tss.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del eval_gt_tss
        gc.collect()
        joblib.dump(eval_gt_full, f"pickle/{chr_name}_eval_gt_full.gz", compress="lz4")
        # pickle.dump(eval_gt_full, open(f"pickle/{chr_name}_eval_gt_full.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        test_seq = joblib.load(f"pickle/{chr_name}_seq.gz")
        eval_gt = joblib.load(f"pickle/{chr_name}_eval_gt.gz")
        eval_gt_tss = joblib.load(f"pickle/{chr_name}_eval_gt_tss.gz")

    start_val = {}
    tracks_for_bed = {"scEnd5": []}# {"CAGE":[], "scEnd5":[], "scATAC":[]}
    bed_num = 5
    track_inds_bed = []
    for t, track in enumerate(eval_track_names):
        type = track[:track.find(".")]
        if type not in tracks_for_bed.keys():
            continue
        if len(tracks_for_bed[type]) < bed_num:
            track_inds_bed.append(t)
            tracks_for_bed[type].append(track)
        else:
            continue

    final_pred = {}
    for i in range(len(eval_infos)):
        final_pred[eval_infos[i][2]] = {}

    for w in range(0, len(test_seq), w_step):
        print(w, end=" ")
        p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
        if len(hic_keys) > 0:
            p2 = p1[0][:, :, p.mid_bin - 1] + p1[0][:, :, p.mid_bin] + p1[0][:, :, p.mid_bin + 1]
            if w == 0:
                predictions = p2
                predictions_full = p1[0]
                predictions_hic = p1[1]
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float32)
                if w / w_step < full_preds_steps_num:
                    predictions_full = np.concatenate((predictions_full, p1[0]), dtype=np.float16)
                    predictions_hic = np.concatenate((predictions_hic, p1[1]), dtype=np.float16)
        else:
            p2 = p1[:, :, p.mid_bin - 1] + p1[:, :, p.mid_bin] + p1[:, :, p.mid_bin + 1]
            if w == 0:
                predictions = p2
                predictions_full = p1
                print(f"MSE: {np.square(np.subtract(predictions_full, eval_gt_full)).mean()}")
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float32)
                if w / w_step < full_preds_steps_num:
                    predictions_full = np.concatenate((predictions_full, p1), dtype=np.float16)

        if len(hic_keys) > 0:
            predictions_for_bed = p1[0]
        else:
            predictions_for_bed = p1

        print(" -bed ", end="")
        for c, locus in enumerate(predictions_for_bed):
            ind = w + c
            mid = eval_infos[ind][1] - p.half_num_regions * p.bin_size - (eval_infos[ind][1] % p.bin_size)
            for b in range(p.num_regions):
                start = mid + b * p.bin_size
                for t in track_inds_bed:
                    track = eval_track_names[t]
                    start_val.setdefault(track, {}).setdefault(start, []).append(locus[t][b])
        print(" bed- ", end="")
        p1 = None
        p2 = None
        predictions_for_bed = None
        gc.collect()

    p_gene_set = []
    final_pred_tss = {}
    for i in range(len(eval_infos)):
        if not eval_infos[i][5]:
            p_gene_set.append(eval_infos[i][2])
        for it, track in enumerate(eval_track_names):
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
        for gene in p_gene_set:
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

    print("Saving bed files")
    for track in start_val.keys():
        for start in start_val[track].keys():
            start_val[track][start] = np.mean(start_val[track][start]) # MAX can be better on the test set!
        # with open("bed_output/" + chr_name + "_" + track + ".bedGraph", 'w+') as f:
        #     for start in sorted(start_val[track].keys()):
        #         f.write(f"{chr_name}\t{start}\t{start+p.bin_size}\t{start_val[track][start]}")
        #         f.write("\n")

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
        track_perf[track] = sc
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

    print("Across tracks (TSS) [Averaged across evaluations]")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}
    for track in start_val.keys():
        type = track[:track.find(".")]
        a = eval_gt_tss[track]
        # b = final_pred_tss[track]
        b = []
        for info in eval_infos:
            mid = info[1] - (info[1] % p.bin_size)
            val = start_val[track][mid - p.bin_size] + start_val[track][mid] + start_val[track][mid + p.bin_size]
            b.append(val)
        a = np.nan_to_num(a, neginf=0, posinf=0)
        b = np.nan_to_num(b, neginf=0, posinf=0)
        pc = stats.pearsonr(a, b)[0]
        sc = stats.spearmanr(a, b)[0]
        track_perf[track] = sc
        if pc is not None and sc is not None:
            corrs_p.setdefault(type, []).append((pc, track))
            corrs_s.setdefault(type, []).append((sc, track))
            all_track_spearman[track] = stats.spearmanr(a, b)[0]

    for track_type in corrs_p.keys():
        type_pcc = [i[0] for i in corrs_p[track_type]]
        print(f"{track_type} correlation : {np.mean(type_pcc)}"
              f" {np.mean([i[0] for i in corrs_s[track_type]])} {len(type_pcc)}")

    if should_draw:
        hic_output = []
        for hi, key in enumerate(hic_keys):
            hdf = joblib.load(p.parsed_hic_folder + key)
            ni = 0
            for i, info in enumerate(eval_infos):
                hd = hdf[info[0]]
                hic_mat = np.zeros((p.num_hic_bins, p.num_hic_bins))
                start_hic = int(info[1] - (info[1] % p.bin_size) - p.half_size_hic)
                end_hic = start_hic + 2 * p.half_size_hic
                start_row = hd['locus1'].searchsorted(start_hic - p.hic_bin_size, side='left')
                end_row = hd['locus1'].searchsorted(end_hic, side='right')
                hd = hd.iloc[start_row:end_row]
                # convert start of the input region to the bin number
                start_hic = int(start_hic / p.hic_bin_size)
                # subtract start bin from the binned entries in the range [start_row : end_row]
                l1 = (np.floor(hd["locus1"].values / p.hic_bin_size) - start_hic).astype(int)
                l2 = (np.floor(hd["locus2"].values / p.hic_bin_size) - start_hic).astype(int)
                hic_score = hd["score"].values
                # drop contacts with regions outside the [start_row : end_row] range
                lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
                l1 = l1[lix]
                l2 = l2[lix]
                hic_score = hic_score[lix]
                hic_mat[l1, l2] += hic_score
                hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
                if hi == 0:
                    hic_output.append([])
                hic_output[ni].append(hic_mat)
                ni += 1
            del hd
            del hdf
            gc.collect()

        draw_arguments = [p, eval_track_names, track_perf, predictions_full, eval_gt_full,
                        test_seq, eval_infos, hic_keys,
                        predictions_hic, hic_output,
                        f"{p.figures_folder}/tracks/epoch_{current_epoch}"]
        joblib.dump(draw_arguments, "draw", compress=3)

        viz.draw_tracks(*draw_arguments)

        viz.draw_regplots(eval_track_names, track_perf, final_pred, eval_gt,
                          f"{p.figures_folder}/plots/epoch_{current_epoch}")
        viz.draw_attribution()

    return return_result


def change_seq(x):
    return cm.rev_comp(x)