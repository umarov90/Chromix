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
import pickle
import itertools as it
import traceback
from multiprocessing import Pool, Manager
import multiprocessing as mp
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import cooler
import math
from numba import jit


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


def parse_tracks_sc(p, infos):
    if Path(f"{p.pickle_folder}track_names_sc.gz").is_file():
        track_names_final = joblib.load(f"{p.pickle_folder}track_names_sc.gz")
    else:
        ga = joblib.load(f"{p.pickle_folder}hg38_ga.gz")
        meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
        track_names = []
        for filename in os.listdir(p.tracks_folder_sc):
            track_names.append(filename)
        print(f"{len(track_names)}")
        non_zero_counts = []
        step_size = 100
        q = mp.Queue()
        ps = []
        parsed_tracks = []
        for i in range(len(track_names)):
            new_track_name = track_names[i]
            if new_track_name.endswith(".64nt.bed.gz"):
                new_track_name = new_track_name[:-len(".64nt.bed.gz")]
            new_track_name += ".parsed"
            parsed_tracks.append(new_track_name)
            proc = mp.Process(target=parse_some_tracks,
                              args=(
                              q, p, [track_names[i]], ga, p.bin_size, p.tracks_folder_sc, new_track_name, meta, infos,))
            proc.start()
            ps.append(proc)
            if len(ps) > 100:
                for proc in ps:
                    non_zero_counts.append(q.get())
                ps = []

        if len(ps) > 0:
            for proc in ps:
                non_zero_counts.append(q.get())
        print(f"Non zero mean {np.mean(non_zero_counts)} median {np.median(non_zero_counts)}")

        # Some tracks will be skipped, check what was parsed
        track_names_final = []
        for track in parsed_tracks:
            new_path = p.parsed_tracks_folder + track
            if Path(new_path).exists():
                track_names_final.append(track)
            else:
                print(track)
        print(f"Final tracks {len(track_names_final)}")
        joblib.dump(track_names_final, f"{p.pickle_folder}track_names_sc.gz", compress="lz4")
    return track_names_final


def parse_tracks(p, infos):
    track_names_col = {}
    meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
    q = mp.Queue()
    for specie in p.species:
        if Path(f"{p.pickle_folder}track_names_{specie}.gz").is_file():
            track_names_final = joblib.load(f"{p.pickle_folder}track_names_{specie}.gz")
        else:
            ga = joblib.load(f"{p.pickle_folder}{specie}_ga.gz")
            tracks_folder = p.tracks_folder + specie + "/"
            skip = []
            parsed_tracks = []
            if specie in ["hg38", "mm10"]:
                # Parse encode averaging replicates
                encode_tracks = pd.read_csv(f"data/{specie}.parsed.meta_data.tsv", sep="\t")
                ge = encode_tracks.groupby(["Assay", "Experiment_target", "Biosample_type", "Biosample_term_id"])
                al = ge.agg({'tag': lambda x: list(x)})["tag"].tolist()
                ps = []
                for sl in al:
                    new_track_name = ".".join(sl[0].split(".")[:-2]) + ".parsed"
                    parsed_tracks.append(new_track_name)
                    for i in range(len(sl)):
                        sl[i] += ".64nt.bed.gz"
                    skip += sl
                    proc = mp.Process(target=parse_some_tracks,
                                      args=(q, p, sl, ga, p.bin_size, tracks_folder, new_track_name, meta, infos))
                    proc.start()
                    ps.append(proc)
                    if len(ps) >= mp.cpu_count():
                        for proc in ps:
                            q.get()
                        ps = []
                if len(ps) > 0:
                    for proc in ps: 
                        q.get()

            # Read all other tracks averaging based on track name
            track_names = []
            non_zero_counts = []
            for filename in os.listdir(tracks_folder):
                fn = tracks_folder + filename
                size = os.path.getsize(fn)
                if size > 20000 and filename not in skip:
                    track_names.append(filename)
            print(f"{specie} {len(track_names)}")
            kl, al = group_names_with_rep(track_names)
            ps = []
            for i in range(len(kl)):
                sl = al[i]
                if kl[i].endswith(".64nt.bed.gz"):
                    kl[i] = kl[i][:-len(".64nt.bed.gz")]
                new_track_name = kl[i] + ".parsed"
                parsed_tracks.append(new_track_name)
                proc = mp.Process(target=parse_some_tracks,
                                  args=(q, p, sl, ga, p.bin_size, tracks_folder, new_track_name, meta, infos))
                proc.start()
                ps.append(proc)
                if len(ps) >= 4 * mp.cpu_count():
                    for proc in ps:
                        non_zero_counts.append(q.get())
                    ps = []
            if len(ps) > 0:
                for proc in ps:
                    non_zero_counts.append(q.get())

            non_zero_counts = np.asarray(non_zero_counts)
            non_zero_counts = non_zero_counts[non_zero_counts != 0]
            print(f"Non zero mean {np.mean(non_zero_counts)} median {np.median(non_zero_counts)}")
            # Some tracks will be skipped, check what was parsed
            track_names_final = []
            for track in parsed_tracks:
                new_path = p.parsed_tracks_folder + track
                if Path(new_path).exists():
                    track_names_final.append(track)
            print(f"{specie} final tracks {len(track_names_final)}")
            joblib.dump(track_names_final, f"{p.pickle_folder}track_names_{specie}.gz", compress="lz4")
        track_names_col[specie] = track_names_final

    joblib.dump(track_names_col, f"{p.pickle_folder}track_names_col.gz", compress="lz4")
    return track_names_col


def group_names_with_rep(names):
    result = {}
    for name in names:
        prefix = re.split("_rep\d+(?:[^\.])*\.", name)[0]
        if prefix in result:
            result[prefix].append(name)
        else:
            result[prefix] = [name]
    return list(result.keys()), list(result.values())


def parse_some_tracks(q, p, some_tracks, ga, bin_size, tracks_folder, new_track_name, meta, infos):
    gast = copy.deepcopy(ga)
    new_path = p.parsed_tracks_folder + new_track_name
    count = 0
    meta_row = meta.loc[meta['file_name'] == new_track_name]
    if len(meta_row) > 0:
        meta_row = meta_row.iloc[0]
    else:
        print("no meta " + new_track_name)
        q.put(0)
        return None
    for track in some_tracks:
        try:
            fn = tracks_folder + track
            df = pd.read_csv(fn, delim_whitespace=True, header=None, index_col=False)
            if len(df.columns) == 4:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "score"})
                df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
            elif len(df.columns) == 6:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "id", 4: "score", 5: "strand"})
                df["mid"] = df["start"] // bin_size
            else:
                df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "id", 4: "score", 5: "strand", 6: "tss"})
                df["mid"] = df["tss"] // bin_size
            total_reads = df['score'].sum()
            # if meta_row["value"] == "RNA" and total_reads < 100000:
            #     print(f"Skipping: {track} {total_reads}")
            #     continue
            df["mid"] = df["mid"].astype(int)
            df["score"] = df["score"].astype(np.float32)
            df = df[["chr", "start", "end", "score", "mid"]]
            chrd = list(df["chr"].unique())
            grouped_scores = df.groupby("chr").agg(list)
            for key, val in gast.items():
                if key not in chrd:
                    continue
                pos, score = grouped_scores.loc[key, ["mid", "score"]]
                add_fast(gast[key], np.asarray(pos), np.asarray(score, dtype=np.float32))
            print(f"Parsed {track}.")
            count += 1
        except Exception as exc:
            print(exc)
            traceback.print_exc()
            print("\n\n\nCould not parse! " + track)
    count_non_zero = 0
    if count > 0:
        non_zero_all = []
        for key in gast.keys():
            gast[key] = gast[key] / count
            non_zero = gast[key][np.where(gast[key] != 0)]
            non_zero_all.append(non_zero)
        non_zero_all = np.concatenate(non_zero_all, axis=0, dtype=np.float32)
        if meta_row["value"] == "RNA":
            tss_vals = []
            for info in infos:
                binx = info[1] // p.bin_size
                tss_vals.append(max(gast[info[0]][binx - 1], gast[info[0]][binx], gast[info[0]][binx + 1]))
            qnts = np.quantile(tss_vals, [0.90, 0.98])
            count_non_zero = len([element for element in tss_vals if element != 0])
        else:
            qnts = np.quantile(non_zero_all, [0.99, 0.99])
        for key in gast.keys():
            if meta_row["value"] == "conservation" or meta_row["unit"] == "logcpm":
                pass
            elif meta_row["value"] == "RNA":
                if meta_row["technology"] == "scAtlas":
                    # gast[key] = np.minimum(gast[key], qnts[1] + np.sqrt(np.maximum(0, gast[key] - qnts[1])))
                    gast[key] = np.log10(gast[key] + 1)
                    # gast[key] = gast[key] / np.log(qnts[1] + 1)
                else:
                    # gast[key] = np.minimum(gast[key], qnts[0] + np.sqrt(np.maximum(0, gast[key] - qnts[0])))
                    gast[key] = np.log10(gast[key] + 1)
                    # gast[key] = gast[key] / np.log(qnts[0] + 1)
            else:
                # gast[key] = np.minimum(gast[key], qnts[0] + np.sqrt(np.maximum(0, gast[key] - qnts[0])))
                # if meta_row["unit"] == "read_depth"
                gast[key] = np.log10(gast[key] + 1)
                # gast[key] = gast[key] / np.log(qnts[0] + 1)
            gast[key][~np.isfinite(gast[key])] = 0
            gast[key] = np.round(gast[key], 3)
            gast[key] = gast[key].astype(np.float16)

        joblib.dump(gast, new_path, compress=3)
    q.put(count_non_zero)


@jit(nopython=True)
def add_fast(a, pos, score):
    for i in range(len(pos)):
        a[pos[i]] = a[pos[i]] + score[i]


def parse_sequences(p):
    if Path(f"{p.pickle_folder}train_info.gz").is_file():
        test_info = joblib.load(f"{p.pickle_folder}test_info.gz")
        train_info = joblib.load(f"{p.pickle_folder}train_info.gz")
        valid_info = joblib.load(f"{p.pickle_folder}valid_info.gz")
        protein_coding = joblib.load(f"{p.pickle_folder}protein_coding.gz")
    else:
        gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
        train_tss = pd.read_csv("data/final_train_tss.bed", sep="\t", index_col=False,
                                names=["chrom", "start", "end", "geneID", "score", "strand"])
        test_tss = pd.read_csv("data/final_test_tss.bed", sep="\t", index_col=False,
                               names=["chrom", "start", "end", "geneID", "score", "strand"])
        valid_tss = pd.read_csv("data/final_valid_tss.bed", sep="\t", index_col=False,
                                names=["chrom", "start", "end", "geneID", "score", "strand"])
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

        for sp in p.species:
            if Path(f"{p.pickle_folder}{sp}_regions.gz").is_file():
                continue
            print(sp)
            genome, ga = cm.parse_genome(f"data/species/{sp}/genome.fa", p.bin_size)
            chromosomes = [chrom for chrom in genome.keys() if re.match("chr([0-9]*|X)$", chrom)]
            regions = []
            for chrom in chromosomes:
                for i in range(0, len(genome[chrom]), 50000):
                    regions.append([chrom, i])

            exclude = pd.read_csv(f"data/species/{sp}/exclude.bed", sep="\t", index_col=False,
                                  names=["chrom", "start", "end", "geneID", "score", "strand"])
            blacklist_dict = {}
            if sp == "hg38":
                print("Excluding enformer test and valid set and encode blacklist")
                encode_blacklist = pd.read_csv(f"data/hg38-blacklist.v2.bed", sep="\t", index_col=False,
                                               names=["chrom", "start", "end", "reason"])
                # for index, row in encode_blacklist.iterrows():
                #     blacklist_dict.setdefault(row["chrom"], []).append([int(row["start"]), int(row["end"])])
                enformer_valid = pd.read_csv(f"data/human_valid.bed", sep="\t", index_col=False,
                                             names=["chrom", "start", "end", "type"])
                enformer_test = pd.read_csv(f"data/human_test.bed", sep="\t", index_col=False,
                                            names=["chrom", "start", "end", "type"])
                for index, row in enformer_valid.iterrows():
                    blacklist_dict.setdefault(row["chrom"], []).append([int(row["start"]), int(row["end"])])
                for index, row in enformer_test.iterrows():
                    blacklist_dict.setdefault(row["chrom"], []).append([int(row["start"]), int(row["end"])])
            else:
                for index, row in exclude.iterrows():
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

            joblib.dump(one_hot, f"{p.pickle_folder}{sp}_one_hot.gz", compress="lz4")
            joblib.dump(ga, f"{p.pickle_folder}{sp}_ga.gz", compress=3)
            joblib.dump(regions, f"{p.pickle_folder}{sp}_regions.gz", compress="lz4")
    return train_info, valid_info, test_info, protein_coding


def load_data(mp_q, p, tracks, scores, t, t_end):
    scores_after_loading = np.zeros((len(scores), t_end - t, int(scores[0][2]) - int(scores[0][1])), dtype=np.float16)
    for i, track_name in enumerate(tracks):
        parsed_track = joblib.load(p.parsed_tracks_folder + track_name)
        for j in range(len(scores)):
            scores_after_loading[j, i] = parsed_track[scores[j][0]][int(scores[j][1]):int(scores[j][2])].copy()
    mp_q.put((t, scores_after_loading))


def load_data_sum(mp_q, p, tracks, scores, t, t_end):
    scores_after_loading = np.zeros((len(scores), t_end - t), dtype=np.float16)
    for i, track_name in enumerate(tracks):
        parsed_track = joblib.load(p.parsed_tracks_folder + track_name)
        for j in range(len(scores)):
            # 3 or 6 bins?
            pt = parsed_track[scores[j][0]]
            mid = int(scores[j][1])
            scores_after_loading[j, i] = pt[mid]  # pt[mid - 1] + pt[mid] + pt[mid + 1]
    mp_q.put((t, scores_after_loading))


def par_load_data(output_scores_info, tracks, p):
    mp_q = mp.Queue()
    ps = []
    start = 0
    nproc = min(4 * mp.cpu_count(), len(tracks))
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

    all_scores = []
    for load_proc in ps:
        all_scores.append(mp_q.get())
    all_scores = sorted(all_scores, key=lambda x: x[0])
    all_scores = [tp[1] for tp in all_scores]
    all_scores = np.concatenate(all_scores, axis=1, dtype=np.float16)
    if len(output_scores_info[0]) == 3:
        all_scores = all_scores.swapaxes(1, 2)
    return all_scores


def par_load_hic_data(hic_tracks, p, picked_regions, half):
    mp_q = mp.Queue()
    ps = []

    nproc = min(4 * mp.cpu_count(), len(picked_regions))
    print(nproc)
    step_size = len(picked_regions) // nproc
    end = len(picked_regions)

    for t in range(0, end, step_size):
        t_end = min(t + step_size, end)
        load_proc = mp.Process(target=load_hic_data,
                               args=(mp_q, hic_tracks, picked_regions[t:t_end], t, p, half,))
        load_proc.start()
        ps.append(load_proc)

    all_scores = []
    for load_proc in ps:
        all_scores.append(mp_q.get())
    all_scores = sorted(all_scores, key=lambda x: x[0])
    all_scores = [tp[1] for tp in all_scores]
    all_scores = np.concatenate(all_scores, axis=0, dtype=np.float16)
    all_scores = all_scores.swapaxes(1, 2)
    return all_scores


def load_hic_data(mp_q, hic_tracks, picked_regions, t, p, half):
    hic_data = []
    for i in range(len(picked_regions)):
        hic_data.append([])

    for hic in hic_tracks:
        if hic.endswith("mcool"):
            c = cooler.Cooler(p.hic_folder + hic + "::resolutions/2000")
        else:
            c = cooler.Cooler(p.hic_folder + hic)
        for i, info in enumerate(picked_regions):
            start_hic = info[1] - p.half_size_hic
            start_hic = start_hic - start_hic % p.hic_bin_size
            end_hic = start_hic + 2 * p.half_size_hic
            hic_mat = c.matrix(balance=True).fetch(f'{info[0]}:{start_hic}-{end_hic}')
            hic_mat[np.isnan(hic_mat)] = 0
            hic_mat = hic_mat - np.diag(np.diag(hic_mat, k=1), k=1) - np.diag(np.diag(hic_mat, k=-1), k=-1) - np.diag(
                np.diag(hic_mat))
            hic_mat = gaussian_filter(hic_mat, sigma=1)
            if t + i < half:
                hic_mat = np.rot90(hic_mat, k=2)
            hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=2)]

            # Scale the values
            hic_mat = hic_mat * 100

            hic_data[i].append(hic_mat)
    mp_q.put((t, hic_data))


def par_load_hic_data_one(hic_tracks, p, picked_regions):
    mp_q = mp.Queue()
    ps = []

    nproc = min(4 * mp.cpu_count(), len(picked_regions))
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
    return output_scores


def load_hic_data_one(mp_q, hic_tracks, picked_regions, t, p):
    hic_bin_size = 2000
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
                hic_data[i].append(
                    c.matrix(balance=True, field="count").fetch(f'{info[0]}:{start_hic}-{end_hic}')[0, -1])
            except:
                hic_data[i].append(0)
                print(info)
    hic_data = np.mean(hic_data, axis=-1)
    joblib.dump(hic_data, f"{p.temp_folder}hic_data_one{t}", compress="lz4")
    mp_q.put(None)
