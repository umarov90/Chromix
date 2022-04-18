import os
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


def valid(chunks):
    for chunk in chunks:
        print("1")
        mask = chunk['locus1_chrom'] == chunk['locus2_chrom']
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            break


def parse_hic(folder):
    if Path("pickle/hic_keys.gz").is_file():
        return joblib.load("pickle/hic_keys.gz")
    else:
        hic_keys = []
        directory = "hic"

        for filename in os.listdir(directory):
            fn = os.path.join(directory, filename)
            t_name = fn.replace("/", "_")
            print(t_name)
            # if t_name not in ["hic_Ery.10kb.intra_chromosomal.interaction_table.tsv",
            #                   "hic_HUVEC.10kb.intra_chromosomal.interaction_table.tsv",
            #                   "hic_Islets.10kb.intra_chromosomal.interaction_table.tsv",
            #                   "hic_SkMC.10kb.intra_chromosomal.interaction_table.tsv"]:
            #     continue
            hic_keys.append(t_name)
            # if Path(folder + t_name + "chr1").is_file():
            #     continue
            with open("hic.txt", "a+") as myfile:
                myfile.write(t_name)
            fields = ["locus1_chrom", "locus2_chrom", "locus1_start", "locus2_start",
                      "pvalue", "logObservedOverExpected"]
            dtypes = {"locus1_chrom": str, "locus2_chrom": str, "locus1_start": int, "locus2_start": int,
                      "pvalue": str, "logObservedOverExpected": float}
            chunksize = 10 ** 8
            chunks = pd.read_csv(fn, sep="\t", index_col=False, usecols=fields,
                                 dtype=dtypes, chunksize=chunksize, low_memory=True)
            df = pd.concat(valid(chunks))
            # df = pd.read_csv(fn, sep="\t", index_col=False, usecols=fields, dtype=dtypes, low_memory=True)
            df['pvalue'] = pd.to_numeric(df['pvalue'], errors='raise')
            # df['pvalue'].fillna(0, inplace=True)
            print(len(df))
            # No inter-chromosome connections are considered
            df.drop(df[df['locus1_chrom'] != df['locus2_chrom']].index, inplace=True)
            print(len(df))
            df.drop(['locus2_chrom'], axis=1, inplace=True)
            df.drop(df[df['locus1_start'] - df['locus2_start'] > 420000].index, inplace=True)
            print(len(df))

            df["pvalue"] = -1 * np.log(df["pvalue"])
            m = df.loc[df['pvalue'] != np.inf, 'pvalue'].max()
            print("P Max is: " + str(m))
            df['pvalue'].replace(np.inf, m, inplace=True)
            df['pvalue'].clip(upper=100, inplace=True)
            df["score"] = df["pvalue"] / df["pvalue"].max()
            # df["score"] = df["logObservedOverExpected"] / df["logObservedOverExpected"].max()

            df.drop(["pvalue"], axis=1, inplace=True)
            chrd = list(df["locus1_chrom"].unique())
            for chr in chrd:
                joblib.dump(df.loc[df['locus1_chrom'] == chr].sort_values(by=['locus1_start']),
                            folder + t_name + chr, compress=3)
            print(t_name)
            with open("hic.txt", "a+") as myfile:
                myfile.write(t_name)
            del df
            gc.collect()

        joblib.dump(hic_keys, "pickle/hic_keys.gz", compress=3)
        chromosomes = ["chrX", "chrY"]
        for i in range(1, 23):
            chromosomes.append("chr" + str(i))
        for key in hic_keys:
            print(key)
            hdf = {}
            for chr in chromosomes:
                try:
                    hdf[chr] = joblib.load(folder + key + chr)
                except:
                    pass
            joblib.dump(hdf, folder + key, compress=3)
            print(key)
        return hic_keys


def parse_tracks(gas, bin_size, chromosomes, tracks_folders):
    track_names_col = []
    for i in range(len(tracks_folders)):
        ga = gas[i]
        tracks_folder = tracks_folders[i]
        track_names = []
        for filename in os.listdir(tracks_folder):
            if filename.endswith(".gz"):
                track = filename[:-len(".100nt.bed.gz")]
                fn = tracks_folder + f"{track}.100nt.bed.gz"
                size = os.path.getsize(fn)
                if size > 2 * 512000 or track.startswith("sc"):
                    track_names.append(track)

        print(f"gas {len(track_names)}")

        step_size = 50
        q = mp.Queue()
        ps = []
        start = 0
        nproc = 28
        end = len(track_names)
        for t in range(start, end, step_size):
            t_end = min(t+step_size, end)
            sub_tracks = track_names[t:t_end]
            p = mp.Process(target=parse_some_tracks,
                           args=(q, sub_tracks, ga, bin_size, chromosomes, tracks_folder,))
            p.start()
            ps.append(p)
            if len(ps) >= nproc:
                for p in ps:
                    p.join()
                print(q.get())
                ps = []

        if len(ps) > 0:
            for p in ps:
                p.join()
            print(q.get())

        track_names_col.append(track_names)

    joblib.dump(track_names_col, "pickle/track_names_col.gz", compress=3)
    return track_names_col


def parse_some_tracks(q, some_tracks, ga, bin_size, chromosomes, tracks_folder):
    for track in some_tracks:
        try:
            fn = tracks_folder + f"{track}.100nt.bed.gz"
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
            # all_vals = None
            for key in gast.keys():
                if "scEnd5" in track:
                    gast[key] = np.log10(np.exp(gast[key]))
                else:
                    gast[key] = np.log10(gast[key] + 1)

                if key in chromosomes:
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
                gast[key] = gast[key] / max_val  # np.clip(gast[key], 0, scale_val) / scale_val
                gast[key] = gaussian_filter(gast[key], sigma=0.5)
            joblib.dump(gast, main.p.parsed_tracks_folder + track, compress="lz4")
            # pickle.dump(gast, open(main.p.parsed_tracks_folder + track, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Parsed {track}. Max value: {max_val}.")
        except Exception as exc:
            print(exc)
            traceback.print_exc()
            print("\n\n\nCould not parse! " + track)
    q.put(None)


def get_sequences(bin_size):
    if Path("pickle/mouse_ga.gz").is_file():
        human_genome = joblib.load("pickle/human_genome.gz")
        human_ga = joblib.load("pickle/human_ga.gz")
        mouse_genome = joblib.load("pickle/mouse_genome.gz")
        mouse_ga = joblib.load("pickle/mouse_ga.gz")
    else:
        human_genome, human_ga = cm.parse_genome("data/hg38.fa", bin_size)
        mouse_genome, mouse_ga = cm.parse_genome("data/mm10.fa", bin_size)
        joblib.dump(human_genome, "pickle/genome.gz", compress=3)
        joblib.dump(human_ga, "pickle/ga.gz", compress=3)

    gas = [human_ga, mouse_ga]

    if Path("pickle/train_info.gz").is_file():
        test_info = joblib.load("pickle/test_info.gz")
        train_info = joblib.load("pickle/train_info.gz")
        human_exclude_dict = joblib.load("pickle/tss_loc.gz")
        protein_coding = joblib.load("pickle/protein_coding.gz")
        one_hot_human = joblib.load("pickle/one_hot_human.gz")
        one_hot_mouse = joblib.load("pickle/one_hot_mouse.gz")
        mouse_regions = joblib.load("pickle/mouse_regions.gz")
        human_regions = joblib.load("pickle/human_regions.gz")
    else:
        gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
        train_tss = pd.read_csv("data/final_train_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        test_tss = pd.read_csv("data/final_test_tss.bed", sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        protein_coding = []
        test_info = []
        human_exclude_dict = {}
        for index, row in test_tss.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
            # if gene_type != "protein_coding":
            #     continue
            human_exclude_dict.setdefault(row["chrom"], []).append(pos)
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

        human_regions = pd.read_csv("data/human.genome.windows.bed", sep="\t",
                                    index_col=False, names=["chrom", "start", "end"])
        human_regions["mid"] = (human_regions["start"] + (human_regions["end"] - human_regions["start"]) / 2)
        human_chromosomes = human_regions['chrom'].unique().tolist()
        human_regions = human_regions.astype({"mid": int})
        human_regions = human_regions[['chrom', 'mid']].values.tolist()

        one_hot_human = {}
        for chromosome in human_chromosomes:
            print(chromosome)
            one_hot_human[chromosome] = cm.encode_seq(human_genome[chromosome])
            tss_layer = np.zeros((len(one_hot_human[chromosome]), 1)).astype(bool)
            print(len(one_hot_human[chromosome]))
            if chromosome in human_exclude_dict.keys():
                for tss in human_exclude_dict[chromosome]:
                    tss_layer[tss, 0] = True
            print(f"{chromosome}: {np.sum(tss_layer)}")
            one_hot_human[chromosome] = np.hstack([one_hot_human[chromosome], tss_layer])

        mouse_regions = pd.read_csv("data/mouse.genome.windows.bed", sep="\t",
                                    index_col=False, names=["chrom", "start", "end"])
        mouse_regions["mid"] = (mouse_regions["start"] + (mouse_regions["end"] - mouse_regions["start"]) / 2)
        mouse_regions = mouse_regions.astype({"mid": int})
        mouse_chromosomes = mouse_regions['chrom'].unique().tolist()
        mouse_regions = mouse_regions[['chrom', 'mid']].values.tolist()

        mouse_exclude = pd.read_csv("data/mouse_exclude.bed", sep="\t", index_col=False,
                               names=["chrom", "start", "end", "geneID", "score", "strand"])
        mouse_exclude_dict = {}
        for index, row in mouse_exclude.iterrows():
            pos = int(row["start"])
            mouse_exclude_dict.setdefault(row["chrom"], []).append(pos)

        one_hot_mouse = {}
        for chromosome in mouse_chromosomes:
            print(chromosome)
            one_hot_mouse[chromosome] = cm.encode_seq(mouse_genome[chromosome])
            tss_layer = np.zeros((len(one_hot_mouse[chromosome]), 1)).astype(bool)
            print(len(one_hot_mouse[chromosome]))
            if chromosome in mouse_exclude_dict.keys():
                for tss in mouse_exclude_dict[chromosome]:
                    tss_layer[tss, 0] = True
            print(f"{chromosome}: {np.sum(tss_layer)}")
            one_hot_mouse[chromosome] = np.hstack([one_hot_mouse[chromosome], tss_layer])

        joblib.dump(one_hot_mouse, "pickle/one_hot_mouse.gz", compress=3)
        joblib.dump(one_hot_human, "pickle/one_hot_human.gz", compress=3)
        joblib.dump(mouse_regions, "pickle/mouse_regions.gz", compress=3)
        joblib.dump(human_regions, "pickle/human_regions.gz", compress=3)
        joblib.dump(test_info, "pickle/test_info.gz", compress=3)
        joblib.dump(train_info, "pickle/train_info.gz", compress=3)
        joblib.dump(human_exclude_dict, "pickle/tss_loc.gz", compress=3)
        joblib.dump(protein_coding, "pickle/protein_coding.gz", compress=3)
        gc.collect()

    one_hots = [one_hot_human, one_hot_mouse]
    training_regions = [human_regions, mouse_regions]

    return gas, one_hots, train_info, test_info, human_exclude_dict, protein_coding, training_regions


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
