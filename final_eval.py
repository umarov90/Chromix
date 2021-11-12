for shift_val in [0]:  # -2 * bin_size, -1 * bin_size, 0, bin_size, 2 * bin_size
    test_seq = []
    for info in eval_infos:
        start = int(info[1] - (info[1] % bin_size) + shift_val - half_size)
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
    for comp in [False]:
        if comp:
            with Pool(4) as p:
                rc_arr = p.map(change_seq, test_seq)
            test_seq = rc_arr
        test_seq = np.asarray(test_seq, dtype=np.float16)
        if comp:
            correction = 1 * int(shift_val / bin_size)
        else:
            correction = -1 * int(shift_val / bin_size)
        print(f"\n{shift_val} {comp} {test_seq.shape} predicting")
        predictions = None
        # predictions_max = None
        for w in range(0, len(test_seq), w_step):
            print(w, end=" ")
            if w != 0 and (w / w_step) % 25 == 0:
                print(" Reloading ")
                gc.collect()
                K.clear_session()
                tf.compat.v1.reset_default_graph()

                with strategy.scope():
                    our_model = tf.keras.models.load_model(model_folder + "/" + model_name + "_no.h5",
                                                           custom_objects={'SAMModel': mo.SAMModel,
                                                                           'PatchEncoder': mo.PatchEncoder})

            p1 = our_model.predict(mo.wrap2(test_seq[w:w + w_step], predict_batch_size))
            # p2 = p1[:, :, mid_bin + correction]
            p2 = p1[:, :, mid_bin - 1 + correction] + p1[:, :, mid_bin + correction] + p1[:, :,
                                                                                       mid_bin + 1 + correction]
            # p_max = np.max(p1, axis=2)
            if w == 0:
                predictions = p2
                # predictions_max = p_max
            else:
                predictions = np.concatenate((predictions, p2), dtype=np.float16)
                # predictions_max = np.concatenate((predictions_max, p_max), dtype=np.float16)
        for i in range(len(eval_infos)):
            for it, ct in enumerate(track_names):
                final_pred[eval_infos[i][2]].setdefault(ct, []).append(predictions[i][it])
                # final_pred_max[eval_infos[i][2]].setdefault(ct, []).append(predictions_max[i][it])
        print(f"{shift_val} {comp} finished")
        del predictions
        gc.collect()