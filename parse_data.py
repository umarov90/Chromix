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
import cooler


def parse_hic(p):
    if Path(f"{p.pickle_folder}hic_keys.gz").is_file():
        return joblib.load(f"{p.pickle_folder}hic_keys.gz")
    else:
        hic_keys = []
        directory = "hic"

        for filename in os.listdir(directory):
            fn = os.path.join(directory, filename)
            if not fn.endswith(".mcool"):
                continue
            t_name = filename
            hic_keys.append(t_name)

        return hic_keys


def parse_tracks(p):
    meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
    if Path(f"{p.pickle_folder}track_names.gz").is_file():
        track_names = joblib.load(f"{p.pickle_folder}track_names.gz")
    else:
        ga = joblib.load(f"{p.pickle_folder}ga.gz")
        tracks_folder = p.tracks_folder
        track_names = []
        for filename in os.listdir(tracks_folder):
            if filename.endswith(".gz") or filename.endswith(".bed"):
                fn = tracks_folder + filename
                size = os.path.getsize(fn)
                if size > 200000 or filename.startswith("sc"):
                    track_names.append(filename)

        print(f"Number of tracks {len(track_names)}")

        step_size = 100
        q = mp.Queue()
        ps = []
        start = 0
        nproc = mp.cpu_count()
        end = len(track_names)
        for t in range(start, end, step_size):
            t_end = min(t+step_size, end)
            sub_tracks = track_names[t:t_end]
            proc = mp.Process(target=parse_some_tracks,
                           args=(q, p, sub_tracks, ga, p.bin_size, tracks_folder,meta,t,))
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
        track_names = []
        for filename in os.listdir(p.parsed_tracks_folder):
            if filename.endswith(".parsed"):
                track_names.append(filename)
        print(f"Final number of tracks {len(track_names)}")
        joblib.dump(track_names, f"{p.pickle_folder}track_names.gz", compress="lz4")
    return track_names


def parse_some_tracks(q, p, some_tracks, ga, bin_size, tracks_folder, meta, t):
    maxes = []
    for track in some_tracks:
        # if Path(main.p.parsed_tracks_folder + track).is_file():
        #     continue
        try:
            fn = tracks_folder + track
            track_name = track
            if track_name.endswith(".64nt.bed.gz"):
                track_name = track_name[:-len(".64nt.bed.gz")] 
            track_name += ".parsed"
            new_path = p.parsed_tracks_folder + track_name
            meta_row = meta.loc[meta['file_name'] == track_name]
            if len(meta_row) > 0:
                meta_row = meta_row.iloc[0]
            else:
                meta_row = None
            # if not ".RNA.ctss" in track:
            #     continue
            # if "scend5" not in track.lower() and not (meta_row is not None and meta_row["value"] == "RNA"):
            #     continue
            # delete in case it already exists
            if Path(new_path).exists():
                Path(new_path).unlink()
            gast = copy.deepcopy(ga)
            df = pd.read_csv(fn, delim_whitespace=True, header=None, index_col=False)
            if len(df.columns) == 4:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "score"})
                df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
            else:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "id", 4: "score", 5: "strand", 6: "tss"})
                df["mid"] = df["tss"] // bin_size
            total_reads = df['score'].sum()
            # print(df.iloc[df['score'].argmax()])
            if meta_row is not None and meta_row["value"] == "RNA" and total_reads < 1000000:
                print(f"Skipping: {track} {total_reads}")
                continue

            df["mid"] = df["mid"].astype(int)
            df["score"] = df["score"].astype(float)
            df = df[["chr", "start", "end", "score", "mid"]]
            chrd = list(df["chr"].unique())

            total_bins = 0
            # group the scores over `key` and gather them in a list
            grouped_scores = df.groupby("chr").agg(list)

            # for each key, value in the dictionary...
            non_zero_all = []
            for key, val in gast.items():
                if key not in chrd:
                    continue
                # first lookup the positions to update and the corresponding scores
                pos, score = grouped_scores.loc[key, ["mid", "score"]]
                # fancy indexing
                gast[key][pos] += score
                total_bins += np.count_nonzero(gast[key])
                non_zero = gast[key][np.where(gast[key] != 0)]
                non_zero_all.append(non_zero)

            non_zero_all = np.concatenate(non_zero_all, axis=0, dtype=np.float32)
            mx = np.max(non_zero_all)
            t1 = 0.01 * mx
            t2 = 0.1 * mx
            # non_zero_all = np.sort(non_zero_all)
            # qnts = np.quantile(non_zero_all, [0.99, 0.998])
            # print(f"{track} {np.max(non_zero_all)}", end=": ")
            # for qnt in qnts:
            #     print(qnt, end=" ")
            # print("")

            # if meta_row is not None and meta_row["value"] == "RNA" and total_bins < 200000:
            #     print(f"Skipping: {track} {total_bins}")
            #     continue

            # if total_bins < 20000:
            #     print(f"Skipping nonrna: {track} {total_bins}")
            #     continue

            max_val = -1
            for key in gast.keys():
                if meta_row["value"] == "conservation":
                    pass
                elif meta_row["value"] == "RNA":
                    gast[key] = np.minimum(gast[key], t1 + np.sqrt(np.maximum(0, gast[key] - t1)))
                else:
                    gast[key] = np.minimum(gast[key], t2 + np.sqrt(np.maximum(0, gast[key] - t2)))
                max_val = max(np.max(gast[key]), max_val)
            for key in gast.keys():
                # if not ".RNA.ctss" in track:
                gast[key] = gast[key] / max_val
                # gast[key] = gaussian_filter(gast[key], sigma=1.0)
                gast[key] = gast[key].astype(np.float16)            
            joblib.dump(gast, new_path, compress="lz4")
            print(f"Parsed {track}. Max value: {max_val}. Sum: {total_reads}. Total bins: {total_bins}. {mx} {t1} {t2}")
            maxes.append(f"{track_name}\t{max_val}")
        except Exception as exc:
            print(exc)
            traceback.print_exc()
            print("\n\n\nCould not parse! " + track)
    with open(f"maxes/{t}", 'w+') as f:
        f.write('\n'.join(maxes))
    q.put(None)


def parse_sequences(p):
    if Path(f"{p.pickle_folder}train_info.gz").is_file():
        test_info = joblib.load(f"{p.pickle_folder}test_info.gz")
        train_info = joblib.load(f"{p.pickle_folder}train_info.gz")
        valid_info = joblib.load(f"{p.pickle_folder}valid_info.gz")
        protein_coding = joblib.load(f"{p.pickle_folder}protein_coding.gz")
    else:
        gene_info = pd.read_csv("data/old_gene.info.tsv", sep="\t", index_col=False)
        train_tss = pd.read_csv("data/final_train_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        test_tss = pd.read_csv("data/final_test_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        valid_tss = pd.read_csv("data/final_valid_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        protein_coding = []
        valid_info = []
        for index, row in valid_tss.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
            valid_info.append([row["chrom"], pos, row["geneID"], gene_type,
                              row["strand"], gene_type != "protein_coding", gene_name])
        print(f"Valid set complete {len(valid_info)}")

        test_info = []
        for index, row in test_tss.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
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
            train_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"],
                               gene_type != "protein_coding", gene_name])

        print(f"Training set complete {len(train_info)}")

        joblib.dump(test_info, f"{p.pickle_folder}test_info.gz", compress="lz4")
        joblib.dump(train_info, f"{p.pickle_folder}train_info.gz", compress="lz4")
        joblib.dump(valid_info, f"{p.pickle_folder}valid_info.gz", compress="lz4")
        joblib.dump(protein_coding, f"{p.pickle_folder}protein_coding.gz", compress="lz4")

        genome, ga = cm.parse_genome(f"data/hg38/genome.fa", p.bin_size)
        chromosomes = [chrom for chrom in genome.keys() if re.match("chr([0-9]*|X)$", chrom)]
        regions = []
        for chrom in chromosomes:
            for i in range(0, len(genome[chrom]), 10000):
                regions.append([chrom, i])

        encode_blacklist = pd.read_csv(f"data/hg38-blacklist.v2.bed", sep="\t", index_col=False,
                              names=["chrom", "start", "end", "reason"])
        enformer_valid = pd.read_csv(f"data/enformer_valid.bed", sep="\t", index_col=False,
                              names=["chrom", "start", "end", "type"])
        enformer_test = pd.read_csv(f"data/enformer_test.bed", sep="\t", index_col=False,
                              names=["chrom", "start", "end", "type"])

        blacklist_dict = {}
        for index, row in encode_blacklist.iterrows():
            blacklist_dict.setdefault(row["chrom"], []).append([int(row["start"]), int(row["end"])])
        for index, row in enformer_valid.iterrows():
            blacklist_dict.setdefault(row["chrom"], []).append([int(row["start"]), int(row["end"])])
        for index, row in enformer_test.iterrows():
            blacklist_dict.setdefault(row["chrom"], []).append([int(row["start"]), int(row["end"])])

        one_hot = {}
        for chromosome in chromosomes:
            print(chromosome)
            one_hot[chromosome] = cm.encode_seq(genome[chromosome])
            exclude_layer = np.zeros((len(one_hot[chromosome]), 1)).astype(bool)
            print(len(one_hot[chromosome]))
            if chromosome in blacklist_dict.keys():
                for region in blacklist_dict[chromosome]:
                    exclude_layer[region[0]:region[1], 0] = True
            print(f"{chromosome}: {np.sum(exclude_layer)}")
            one_hot[chromosome] = np.hstack([one_hot[chromosome], exclude_layer])

        joblib.dump(one_hot, f"{p.pickle_folder}one_hot.gz", compress="lz4")
        joblib.dump(ga, f"{p.pickle_folder}ga.gz", compress=3)
        joblib.dump(regions, f"{p.pickle_folder}regions.gz", compress="lz4")
        gc.collect()

    return train_info, valid_info, test_info, protein_coding


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
    output_scores = np.concatenate(output_scores, axis=1, dtype=np.float16)
    gc.collect()
    return output_scores


def par_load_data_temp(output_scores_info, tracks, p):
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
        load_func = load_data_sum_temp
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
        output_scores.append(joblib.load(f"{p.temp_folder}tdata{t}"))
    output_scores = np.concatenate(output_scores, axis=1, dtype=np.float16)
    gc.collect()
    return output_scores

def load_data_sum_temp(mp_q, p, tracks, scores, t, t_end):
    scores_after_loading = np.zeros((len(scores), t_end - t), dtype=np.float16)
    for i, track_name in enumerate(tracks):
        parsed_track = joblib.load(p.parsed_tracks_folder + track_name)
        for j in range(len(scores)):
            # 3 or 6 bins?
            pt = parsed_track[scores[j][0]]
            mid = int(scores[j][1])
            scores_after_loading[j, i] = pt[mid - 1] + pt[mid] + pt[mid + 1]
    joblib.dump(scores_after_loading, f"{p.temp_folder}tdata{t}", compress="lz4")
    mp_q.put(None)


def par_load_hic_data(hic_tracks, p, picked_regions, half):
    mp_q = mp.Queue()
    ps = []

    nproc = min(mp.cpu_count(), len(picked_regions))
    print(nproc)
    step_size = len(picked_regions) // nproc
    end = len(picked_regions)

    for t in range(0, end, step_size):
        t_end = min(t + step_size, end)
        load_proc = mp.Process(target=load_hic_data,
                               args=(mp_q, hic_tracks, picked_regions[t:t_end], t, p, half,))
        load_proc.start()
        ps.append(load_proc)

    for load_proc in ps:
        load_proc.join()
    print(mp_q.get())

    output_scores = []
    for t in range(0, end, step_size):
        output_scores.append(joblib.load(f"{p.temp_folder}hic_data{t}"))
    output_scores = np.concatenate(output_scores, axis=0, dtype=np.float16)
    gc.collect()
    return output_scores


def load_hic_data(mp_q, hic_tracks, picked_regions, t, p, half):
    hic_data = []
    for i in range(len(picked_regions)):
        hic_data.append([])

    for hic in hic_tracks:
        c = cooler.Cooler(p.hic_folder + hic + "::resolutions/5000")
        for i, info in enumerate(picked_regions):
            start_hic = info[1] - p.half_size_hic
            start_hic = start_hic - start_hic % p.hic_bin_size
            end_hic = start_hic + 2 * p.half_size_hic
            hic_mat = c.matrix(balance=True, field="count").fetch(f'{info[0]}:{start_hic}-{end_hic}')
            hic_mat[np.isnan(hic_mat)] = 0
            hic_mat = hic_mat - np.diag(np.diag(hic_mat, k=1), k=1) - np.diag(np.diag(hic_mat, k=-1), k=-1) - np.diag(np.diag(hic_mat))
            hic_mat = gaussian_filter(hic_mat, sigma=1)
            if t + i < half:
                hic_mat = np.rot90(hic_mat, k=2)
            hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=2)]

            # Scale the values
            hic_mat = hic_mat * 10

            hic_data[i].append(hic_mat)
    joblib.dump(hic_data, f"{p.temp_folder}hic_data{t}", compress="lz4")
    mp_q.put(None)


def par_load_hic_data_one(hic_tracks, p, picked_regions):
    mp_q = mp.Queue()
    ps = []

    nproc = min(mp.cpu_count(), len(picked_regions))
    print(nproc)
    step_size = len(picked_regions) // nproc
    end = len(picked_regions)

    for t in range(0, end, step_size):
        t_end = min(t + step_size, end)
        load_proc = mp.Process(target=load_hic_data_one,
                               args=(mp_q, hic_tracks, picked_regions[t:t_end], t, p,))
        load_proc.start()
        ps.append(load_proc)

    for load_proc in ps:
        load_proc.join()
    print(mp_q.get())

    output_scores = []
    for t in range(0, end, step_size):
        output_scores.append(joblib.load(f"{p.temp_folder}hic_data_one{t}"))
    output_scores = np.concatenate(output_scores, axis=0, dtype=np.float16)
    gc.collect()
    return output_scores


def load_hic_data_one(mp_q, hic_tracks, picked_regions, t, p):
    hic_bin_size = 5000
    hic_data = []
    for i in range(len(picked_regions)):
        hic_data.append([])

    for hic in hic_tracks:
        c = cooler.Cooler(p.hic_folder + hic + "::resolutions/" + str(hic_bin_size))
        for i, info in enumerate(picked_regions):
            start_hic = info[1] - hic_bin_size // 2 - info[1] % hic_bin_size
            end_hic = info[2] - hic_bin_size // 2 - info[2] % hic_bin_size
            if start_hic == end_hic:
                end_hic = end_hic + 1
            try:
                hic_data[i].append(c.matrix(balance=True, field="count").fetch(f'{info[0]}:{start_hic}-{end_hic}')[0, -1])
            except:
                hic_data[i].append(0)
                print(info)
    hic_data = np.mean(hic_data, axis=-1)
    joblib.dump(hic_data, f"{p.temp_folder}hic_data_one{t}", compress="lz4")
    mp_q.put(None)