def eval_perf(eval_infos, should_draw, current_epoch, chr_name, head_id):
    import model as mo
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
    if mixed16:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + model_name,
                                               custom_objects={'PatchEncoder': mo.PatchEncoder})
        our_model.get_layer("our_head").set_weights(joblib.load(model_folder + model_name + "_head_" + str(head_id)))
    predict_batch_size = GLOBAL_BATCH_SIZE
    w_step = 16
    full_preds_steps_num = 10
    eval_track_names = heads[head_id]

    if Path(f"pickle/{chr_name}_seq.gz").is_file():
        test_seq = joblib.load(f"pickle/{chr_name}_seq.gz")
        eval_gt = joblib.load(f"pickle/{chr_name}_eval_gt.gz")
        eval_gt_tss = joblib.load(f"pickle/{chr_name}_eval_gt_tss.gz")
        eval_gt_full = joblib.load(f"pickle/{chr_name}_eval_gt_full.gz")
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
                # buf = loaded_tracks[key]
                # parsed_track = joblib.load(buf)
                # buf.seek(0)
                parsed_track = loaded_tracks[key]
            else:
                parsed_track = joblib.load(parsed_tracks_folder + key)
            mids = []
            for j, info in enumerate(eval_infos):
                mid = int(info[1] / bin_size)
                # if mid in mids:
                #     continue
                # mids.append(mid)
                # val = parsed_track[info[0]][mid]
                val = parsed_track[info[0]][mid - 1] + parsed_track[info[0]][mid] + parsed_track[info[0]][mid + 1]
                eval_gt_tss.setdefault(key, []).append(val)
                eval_gt[info[2]].setdefault(key, []).append(val)
                if j < full_preds_steps_num * w_step:
                    start_bin = int(info[1] / bin_size) - half_num_regions
                    extra_bin = start_bin + num_regions - len(parsed_track[info[0]])
                    if start_bin < 0:
                        binned_region = parsed_track[info[0]][0: start_bin + num_regions]
                        binned_region = np.concatenate((np.zeros(-1 * start_bin), binned_region))
                    elif extra_bin > 0:
                        binned_region = parsed_track[info[0]][start_bin: len(parsed_track[info[0]])]
                        binned_region = np.concatenate((binned_region, np.zeros(extra_bin)))
                    else:
                        binned_region = parsed_track[info[0]][start_bin:start_bin + num_regions]
                    eval_gt_full[j].append(binned_region)
            if i == 0:
                print(f"Skipped: {len(eval_infos) - len(mids)}")

        for i, gene in enumerate(eval_gt.keys()):
            if i % 10 == 0:
                print(i, end=" ")
            for track in eval_track_names:
                eval_gt[gene][track] = np.mean(eval_gt[gene][track])
        print("")
        eval_gt_full = np.asarray(eval_gt_full)

        test_seq = []
        starts = []
        for info in eval_infos:
            start = int(info[1] - (info[1] % bin_size) - half_size)
            # if start in starts:
            #     continue
            # starts.append(start)
            extra = start + input_size - len(one_hot[info[0]])
            if start < 0:
                ns = one_hot[info[0]][0:start + input_size]
                ns = np.concatenate((np.zeros((-1 * start, num_features)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, num_features))))
            else:
                ns = one_hot[info[0]][start:start + input_size]
            if len(ns) != input_size:
                print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
            test_seq.append(ns)
        print(f"Skipped: {len(eval_infos) - len(starts)}")
        test_seq = np.asarray(test_seq, dtype=bool)
        print(f"Lengths: {len(test_seq)} {len(eval_gt)} {len(eval_gt_full)}")
        joblib.dump(test_seq, f"pickle/{chr_name}_seq.gz", compress="lz4")
        joblib.dump(eval_gt, f"pickle/{chr_name}_eval_gt.gz", compress="lz4")
        joblib.dump(eval_gt_tss, f"pickle/{chr_name}_eval_gt_tss.gz", compress="lz4")
        joblib.dump(eval_gt_full, f"pickle/{chr_name}_eval_gt_full.gz", compress="lz4")

    final_pred = {}
    for i in range(len(eval_infos)):
        final_pred[eval_infos[i][2]] = {}

    for w in range(0, len(test_seq), w_step):
        print(w, end=" ")
        if w != 0 and (w / w_step) % 40 == 0:
            print(" Reloading ")
            gc.collect()
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
                our_model.get_layer("our_head").set_weights(
                    joblib.load(model_folder + model_name + "_head_" + str(head_id)))

        p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
        # p2 = p1[:, :, mid_bin + correction]
        if len(hic_keys) > 0:
            p2 = p1[0][:, :, mid_bin - 1] + p1[0][:, :, mid_bin] + p1[0][:, :, mid_bin + 1]
            if w == 0:
                predictions = p2
                predictions_full = p1[0]
                predictions_hic = p1[1]
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float32)
                if w / w_step < full_preds_steps_num:
                    predictions_full = np.concatenate((predictions_full, p1[0]), dtype=np.float32)
                    predictions_hic = np.concatenate((predictions_hic, p1[1]), dtype=np.float32)
        else:
            p2 = p1[:, :, mid_bin - 1] + p1[:, :, mid_bin] + p1[:, :, mid_bin + 1]
            # if np.isnan(p2).any() or np.isinf(p2).any():
            #     print("nan predicted")
            #     joblib.dump(test_seq[w:w + w_step], "nan_testseq")
            #     joblib.dump(p2, "nan_testseq")

            if w == 0:
                predictions = p2
                predictions_full = p1
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float32)
                if w / w_step < full_preds_steps_num:
                    predictions_full = np.concatenate((predictions_full, p1), dtype=np.float32)

    final_pred_tss = {}
    for i in range(len(eval_infos)):
        for it, track in enumerate(eval_track_names):
            final_pred[eval_infos[i][2]].setdefault(track, []).append(predictions[i][it])
            final_pred_tss.setdefault(track, []).append(predictions[i][it])

    for i, gene in enumerate(final_pred.keys()):
        if i % 10 == 0:
            print(i, end=" ")
        for track in eval_track_names:
            final_pred[gene][track] = np.mean(final_pred[gene][track])

    print("Across tracks Genes")
    corrs_p = {}
    corrs_s = {}
    all_track_spearman = {}
    track_perf = {}
    p_gene_set = list(set(final_pred.keys()).intersection(set(protein_coding)))
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
    return_result = np.mean([i[0] for i in corrs_s["CAGE"]])

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

    # bed_files = {}
    # for i, locus in enumerate(predictions_full):
    #     locus_start = eval_infos[i][1] - half_num_regions * bin_size - (eval_infos[i][1] % bin_size)
    #     for b in range(num_regions):
    #         start = locus_start + b * bin_size
    #         eval_chr = eval_infos[i][0]
    #         for t, track in enumerate(eval_track_names):
    #             type = track[:track.find(".")]
    #             if type != "CAGE":
    #                 continue
    #             bed_files.setdefault(track, []).append(f"{eval_chr}\t{start}\t{start+bin_size}\t{locus[t][b]}\t.\t.\t.")
    #
    # for key in bed_files.keys():
    #     with open("bed_output/" + key + ".bed", 'w+') as f:
    #         f.write('\n'.join(bed_files[key]))

    print("Across tracks TSS")
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

    if should_draw:
        hic_output = []
        for hi, key in enumerate(hic_keys):
            hdf = joblib.load(parsed_hic_folder + key)
            ni = 0
            for i, info in enumerate(eval_infos):
                hd = hdf[info[0]]
                hic_mat = np.zeros((num_hic_bins, num_hic_bins))
                start_hic = int(info[1] - (info[1] % bin_size) - half_size_hic)
                end_hic = start_hic + 2 * half_size_hic
                start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
                end_row = hd['locus1'].searchsorted(end_hic, side='right')
                hd = hd.iloc[start_row:end_row]
                # convert start of the input region to the bin number
                start_hic = int(start_hic / hic_bin_size)
                # subtract start bin from the binned entries in the range [start_row : end_row]
                l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
                l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
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

        draw_arguments = [eval_track_names, track_perf, predictions_full, eval_gt_full,
                        test_seq, bin_size, num_regions, eval_infos, hic_keys,
                        hic_track_size, predictions_hic, hic_output,
                        num_hic_bins, f"{figures_folder}/tracks/epoch_{current_epoch}"]
        joblib.dump(draw_arguments, "draw", compress=3)

        viz.draw_tracks(eval_track_names, track_perf, predictions_full, eval_gt_full,
                        test_seq, bin_size, num_regions, eval_infos, hic_keys,
                        hic_track_size, predictions_hic, hic_output,
                        num_hic_bins, f"{figures_folder}/tracks/epoch_{current_epoch}")

        viz.draw_regplots(eval_track_names, track_perf, final_pred, eval_gt,
                          f"{figures_folder}/plots/epoch_{current_epoch}")

        viz.draw_attribution()

    return return_result