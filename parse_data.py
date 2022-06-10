import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import gc
import common as cm
import re
import math
import time
import copy
import main
import pickle
import itertools as it
import traceback
from multiprocessing import Pool, Manager
import multiprocessing as mp
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


def parse_hic(p):
    if Path(f"{p.pickle_folder}hic_keys.gz").is_file():
        return joblib.load(f"{p.pickle_folder}hic_keys.gz")
    else:
        hic_keys = []
        directory = "hic"

        for filename in os.listdir(directory):
            try:
                fn = os.path.join(directory, filename)
                if not fn.endswith("10368nt.tsv.gz"):
                    continue
                t_name = filename
                print(t_name)
                # if t_name not in ["hic_Ery.10kb.intra_chromosomal.interaction_table.tsv",
                #                   "hic_HUVEC.10kb.intra_chromosomal.interaction_table.tsv",
                #                   "hic_Islets.10kb.intra_chromosomal.interaction_table.tsv",
                #                   "hic_SkMC.10kb.intra_chromosomal.interaction_table.tsv"]:
                #     continue
                # if Path(folder + t_name + "chr1").is_file():
                #     continue
                df = pd.read_csv(fn, sep="\t", index_col=False)
                df.rename(columns={'hg38_chrom': 'chrom', 'hg38_bin_1_start': 'start1',
                                   'hg38_bin_2_start': 'start2', 'rlogP_max': 'score',
                                   'rlogQ_max': 'score', 'CHiCAGO_score_max': 'score'}, inplace=True)
                if 'score' not in df.columns:
                    print("score not in columns")
                df = df[["chrom", "start1", "start2", "score"]]
                chrd = list(df["chrom"].unique())
                should_continue = False
                for i in range(22):
                    if "chr" + str(i+1) not in chrd:
                        should_continue = True
                        break
                if should_continue or "chrX" not in chrd:
                    print(f"Not all chroms present in {t_name}")
                    continue
                df.drop(df[df['start1'] - df['start2'] > p.input_size].index, inplace=True)
                print(len(df))

                m = df.loc[df['score'] != np.inf, 'score'].max()
                print("P Max is: " + str(m))
                df['score'].replace(np.inf, m, inplace=True)
                df["score"] = df["score"] / m

                for chr in chrd:
                    joblib.dump(df.loc[df['chrom'] == chr].sort_values(by=['start1']),
                                p.parsed_hic_folder + t_name + chr, compress="lz4")
                print(t_name)
                del df
                gc.collect()
                hic_keys.append(t_name)
            except Exception as exc:
                print(exc)
                print(f"!!!!!!!!!!!!!!!{t_name}")

        joblib.dump(hic_keys, f"{p.pickle_folder}hic_keys.gz", compress="lz4")
        chromosomes = ["chrX", "chrY"]
        for i in range(1, 23):
            chromosomes.append("chr" + str(i))
        for key in hic_keys:
            print(key)
            hdf = {}
            for chr in chromosomes:
                try:
                    hdf[chr] = joblib.load(p.parsed_hic_folder + key + chr)
                except:
                    pass
            joblib.dump(hdf, p.parsed_hic_folder + key, compress="lz4")
            print(key)
        return hic_keys


def parse_tracks(p):
    track_names_col = {}
    meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
    for specie in p.species:
        if Path(f"{p.pickle_folder}track_names_{specie}.gz").is_file():
            track_names = joblib.load(f"{p.pickle_folder}track_names_{specie}.gz")
        else:
            ga = joblib.load(f"{p.pickle_folder}{specie}_ga.gz")
            tracks_folder = p.tracks_folder + specie + "/"
            track_names = []
            for filename in os.listdir(tracks_folder):
                if filename.endswith(".gz"):
                    fn = tracks_folder + filename
                    size = os.path.getsize(fn)
                    if size > 200000 or filename.startswith("sc"):
                        track_names.append(filename)

            print(f"{specie} {len(track_names)}")

            step_size = 50
            q = mp.Queue()
            ps = []
            start = 0
            nproc = 28
            end = len(track_names)
            for t in range(start, end, step_size):
                t_end = min(t+step_size, end)
                sub_tracks = track_names[t:t_end]
                # parse_some_tracks(q, sub_tracks, ga, p.bin_size, tracks_folder,meta)
                proc = mp.Process(target=parse_some_tracks,
                               args=(q, sub_tracks, ga, p.bin_size, tracks_folder,meta,))
                proc.start()
                ps.append(proc)
                if len(ps) >= nproc:
                    for proc in ps:
                        proc.join()
                    print(q.get())
                    ps = []

            if len(ps) > 0:
                for proc in ps:
                    proc.join()
                print(q.get())
            joblib.dump(track_names, f"{p.pickle_folder}track_names_{specie}.gz", compress="lz4")
        track_names_col[specie] = track_names

    joblib.dump(track_names_col, f"{p.pickle_folder}track_names_col.gz", compress="lz4")
    return track_names_col


def parse_some_tracks(q, some_tracks, ga, bin_size, tracks_folder, meta):
    for track in some_tracks:
        if Path(main.p.parsed_tracks_folder + track).is_file():
            continue
        try:
            fn = tracks_folder + track
            meta_row = meta.loc[meta['file_name'] == track]
            if len(meta_row) > 0:
                meta_row = meta_row.iloc[0]
            else:
                meta_row = None
            gast = copy.deepcopy(ga)
            df = pd.read_csv(fn, delim_whitespace=True, header=None, index_col=False)
            if len(df.columns) == 4:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "score"})
                df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
            else:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "id", 4: "score", 5: "strand"})
                df["mid"] = df["start"] / bin_size
            df[["start", "end", "score", "mid"]] = df[["start", "end", "score", "mid"]].astype(int)
            df = df[["chr", "start", "end", "score", "mid"]]
            chrd = list(df["chr"].unique())

            # group the scores over `key` and gather them in a list
            grouped_scores = df.groupby("chr").agg(list)

            # for each key, value in the dictionary...
            for key, val in gast.items():
                if key not in chrd:
                    continue
                # first lookup the positions to update and the corresponding scores
                pos, score = grouped_scores.loc[key, ["mid", "score"]]
                # fancy indexing
                gast[key][pos] += score

            max_val = -1
            min_val = 0
            # all_vals = None
            library_size = 0
            for key in gast.keys():
                min_val = min(np.min(gast[key]), min_val)
                library_size += np.sum(gast[key])
            for key in gast.keys():
                if meta_row is None:
                    gast[key] = np.log10(1000000 * (gast[key] / library_size) + 1)
                elif meta_row["technology"] == "scEnd5":
                    gast[key] = np.log10(np.exp(gast[key]))
                elif meta_row["value"] == "RNA":
                    gast[key] = np.log10(1000000 * (gast[key] / library_size) + 1)
                elif meta_row["value"] == "conservation":
                    gast[key] = gast[key] + abs(min_val)
                else:
                    gast[key] = np.log10(gast[key] + 1)
                max_val = max(np.max(gast[key]), max_val)
                # if all_vals is not None:
                #     all_vals = np.concatenate((all_vals, gast[key][tss_loc[key]]))
                # else:
                #     all_vals = gast[key][tss_loc[key]]
            # tss_loc_num = len(all_vals)
            # all_vals = all_vals[all_vals != 0]
            # all_vals.sort()
            # scale_val = all_vals[int(0.95 * len(all_vals))]
            # if scale_val == 0:
            #     print(scale_val)
            for key in gast.keys():
                if not (meta_row is None or meta_row["value"] == "RNA"):
                    gast[key] = gast[key] / max_val  # np.clip(gast[key], 0, scale_val) / scale_val
                gast[key] = gaussian_filter(gast[key], sigma=0.5)
                gast[key] = gast[key].astype(np.float16)
            joblib.dump(gast, main.p.parsed_tracks_folder + track, compress="lz4")
            # pickle.dump(gast, open(main.p.parsed_tracks_folder + track, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Parsed {track}. Max value: {max_val}.")
        except Exception as exc:
            print(exc)
            traceback.print_exc()
            print("\n\n\nCould not parse! " + track)
    q.put(None)


def parse_sequences(p):
    if Path(f"{p.pickle_folder}train_info.gz").is_file():
        test_info = joblib.load(f"{p.pickle_folder}test_info.gz")
        train_info = joblib.load(f"{p.pickle_folder}train_info.gz")
        protein_coding = joblib.load(f"{p.pickle_folder}protein_coding.gz")
    else:
        gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
        train_tss = pd.read_csv("data/final_train_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        test_tss = pd.read_csv("data/final_test_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        protein_coding = []
        test_info = []
        for index, row in test_tss.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
            # if gene_type != "protein_coding":
            #     continue
            test_info.append([row["chrom"], pos, row["geneID"], gene_type,
                              row["strand"], gene_type != "protein_coding", gene_name])

        print(f"Test set complete {len(test_info)}")
        train_info = []
        for index, row in train_tss.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
            # if gene_type != "protein_coding":
            #     continue
            train_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"],
                               gene_type != "protein_coding", gene_name])

        print(f"Training set complete {len(train_info)}")

        joblib.dump(test_info, f"{p.pickle_folder}test_info.gz", compress="lz4")
        joblib.dump(train_info, f"{p.pickle_folder}train_info.gz", compress="lz4")
        joblib.dump(protein_coding, f"{p.pickle_folder}protein_coding.gz", compress="lz4")

    for sp in p.species:
        if Path(f"{p.pickle_folder}{sp}_regions.gz").is_file():
            continue
        print(sp)
        genome, ga = cm.parse_genome(f"data/species/{sp}/genome.fa", p.bin_size)
        chromosomes = [chrom for chrom in genome.keys() if re.match("chr([0-9]*|X)$", chrom)]
        regions = []
        for chrom in chromosomes:
            for i in range(0, len(genome[chrom]), 40000):
                regions.append([chrom, i])

        exclude = pd.read_csv(f"data/species/{sp}/exclude.bed", sep="\t", index_col=False,
                                    names=["chrom", "start", "end", "geneID", "score", "strand"])
        exclude_dict = {}
        for index, row in exclude.iterrows():
            pos = int(row["start"])
            exclude_dict.setdefault(row["chrom"], []).append(pos)

        one_hot = {}
        for chromosome in chromosomes:
            print(chromosome)
            one_hot[chromosome] = cm.encode_seq(genome[chromosome])
            tss_layer = np.zeros((len(one_hot[chromosome]), 1)).astype(bool)
            print(len(one_hot[chromosome]))
            if chromosome in exclude_dict.keys():
                for tss in exclude_dict[chromosome]:
                    tss_layer[tss, 0] = True
            print(f"{chromosome}: {np.sum(tss_layer)}")
            one_hot[chromosome] = np.hstack([one_hot[chromosome], tss_layer])

        joblib.dump(one_hot, f"{p.pickle_folder}{sp}_one_hot.gz", compress="lz4")
        joblib.dump(ga, f"{p.pickle_folder}{sp}_ga.gz", compress=3)
        joblib.dump(regions, f"{p.pickle_folder}{sp}_regions.gz", compress="lz4")
        gc.collect()

    return train_info, test_info, protein_coding


def parse_one_track(ga, bin_size, fn):
    gast = copy.deepcopy(ga)
    dtypes = {"chr": str, "start": int, "end": int, "score": float}
    df = pd.read_csv(fn, delim_whitespace=True, names=["chr", "start", "end", "score"],
                     dtype=dtypes, header=None, index_col=False)

    chrd = list(df["chr"].unique())
    df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
    df = df.astype({"mid": int})

    # group the scores over `key` and gather them in a list
    grouped_scores = df.groupby("chr").agg(list)

    # for each key, value in the dictionary...
    for key, val in gast.items():
        if key not in chrd:
            continue
        # first lookup the positions to update and the corresponding scores
        pos, score = grouped_scores.loc[key, ["mid", "score"]]
        # fancy indexing
        gast[key][pos] += score

    max_val = -1
    for key in gast.keys():
        gast[key] = np.log(gast[key] + 1)
        max_val = max(np.max(gast[key]), max_val)
    for key in gast.keys():
        gast[key] = gast[key] / max_val

    return gast


def load_data(mp_q, p, tracks, scores, t, t_end):
    scores_after_loading = np.zeros((len(scores), t_end - t, p.num_bins), dtype=np.float16)
    for i, track_name in enumerate(tracks):
        parsed_track = joblib.load(p.parsed_tracks_folder + track_name)
        for j in range(len(scores)):
            scores_after_loading[j, i] = parsed_track[scores[j][0]][int(scores[j][1]):int(scores[j][2])].copy()
    joblib.dump(scores_after_loading, f"{p.temp_folder}data{t}", compress="lz4")
    mp_q.put(None)


def load_data_sum(mp_q, p, tracks, scores, t, t_end):
    scores_after_loading = np.zeros((len(scores), t_end - t), dtype=np.float16)
    for i, track_name in enumerate(tracks):
        parsed_track = joblib.load(p.parsed_tracks_folder + track_name)
        for j in range(len(scores)):
            # 3 or 6 bins?
            pt = parsed_track[scores[j][0]]
            mid = int(scores[j][1])
            scores_after_loading[j, i] = pt[mid - 1] + pt[mid] + pt[mid + 1]
    joblib.dump(scores_after_loading, f"{p.temp_folder}data{t}", compress="lz4")
    mp_q.put(None)


def par_load_data(output_scores_info, tracks, p):
    mp_q = mp.Queue()
    ps = []
    start = 0
    nproc = min(mp.cpu_count(), len(tracks))
    print(nproc)
    step_size = len(tracks) // nproc
    end = len(tracks)
    if len(output_scores_info[0]) == 3:
        load_func = load_data
    else:
        load_func = load_data_sum
    for t in range(start, end, step_size):
        t_end = min(t + step_size, end)
        load_proc = mp.Process(target=load_func,
                               args=(mp_q, p, tracks[t:t_end], output_scores_info, t, t_end,))
        load_proc.start()
        ps.append(load_proc)

    for load_proc in ps:
        load_proc.join()
    print(mp_q.get())

    output_scores = []
    for t in range(start, end, step_size):
        output_scores.append(joblib.load(f"{p.temp_folder}data{t}"))

    gc.collect()
    return np.concatenate(output_scores, axis=1, dtype=np.float16)