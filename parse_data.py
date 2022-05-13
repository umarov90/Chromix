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


def parse_hic(folder):
    if Path("pickle/hic_keys.gz").is_file():
        return joblib.load("pickle/hic_keys.gz")
    else:
        hic_keys = []
        directory = "hic"

        for filename in os.listdir(directory):
            try:
                fn = os.path.join(directory, filename)
                if not fn.endswith("10000nt.tsv.gz"):
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
                fields = ["chrom", "start1", "start2", "value"]
                dtypes = {"chrom": str, "start1": int, "start2": int, "value": float}
                df = pd.read_csv(fn, sep="\t", index_col=False, names=fields, dtype=dtypes, header=None)
                chrd = list(df["chrom"].unique())
                should_continue = False
                for i in range(22):
                    if "chr" + str(i+1) not in chrd:
                        should_continue = True
                        break
                if should_continue or "chrX" not in chrd:
                    print(f"Not all chroms present in {t_name}")
                    continue
                df.drop(df[df['start1'] - df['start2'] > 420000].index, inplace=True)
                print(len(df))

                m = df.loc[df['value'] != np.inf, 'value'].max()
                print("P Max is: " + str(m))
                df['value'].replace(np.inf, m, inplace=True)
                # df['value'].clip(upper=100, inplace=True)
                df["score"] = df["value"] / df["value"].max()

                df.drop(["value"], axis=1, inplace=True)

                for chr in chrd:
                    joblib.dump(df.loc[df['chrom'] == chr].sort_values(by=['start1']),
                                folder + t_name + chr, compress="lz4")
                print(t_name)
                del df
                gc.collect()
                hic_keys.append(t_name)
            except Exception as exc:
                print(exc)
                print(f"!!!!!!!!!!!!!!!{t_name}")

        joblib.dump(hic_keys, "pickle/hic_keys.gz", compress="lz4")
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
            joblib.dump(hdf, folder + key, compress="lz4")
            print(key)
        return hic_keys


def parse_tracks(species, bin_size, tracks_folder_parent):
    track_names_col = []
    for i in range(len(species)):
        if Path(f"pickle/track_names_{species[i]}.gz").is_file():
            track_names = joblib.load(f"pickle/track_names_{species[i]}.gz")
        else:
            ga = joblib.load(f"pickle/{species[i]}_ga.gz")
            tracks_folder = tracks_folder_parent + species[i] + "/"
            track_names = []
            for filename in os.listdir(tracks_folder):
                if filename.endswith(".gz"):
                    track = filename[:-len(".100nt.bed.gz")]
                    fn = tracks_folder + f"{track}.100nt.bed.gz"
                    size = os.path.getsize(fn)
                    if size > 200000 or track.startswith("sc"):
                        track_names.append(track)

            print(f"{species[i]} {len(track_names)}")

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
                               args=(q, sub_tracks, ga, bin_size, tracks_folder,))
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
            joblib.dump(track_names, f"pickle/track_names_{species[i]}.gz", compress="lz4")
        track_names_col.append(track_names)

    joblib.dump(track_names_col, "pickle/track_names_col.gz", compress="lz4")
    return track_names_col


def parse_some_tracks(q, some_tracks, ga, bin_size, tracks_folder):
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
                gast[key] = gast[key].astype(np.float32)
            joblib.dump(gast, main.p.parsed_tracks_folder + track, compress="lz4")
            # pickle.dump(gast, open(main.p.parsed_tracks_folder + track, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Parsed {track}. Max value: {max_val}.")
        except Exception as exc:
            print(exc)
            traceback.print_exc()
            print("\n\n\nCould not parse! " + track)
    q.put(None)


def parse_sequences(species, bin_size):
    if Path("pickle/train_info.gz").is_file():
        test_info = joblib.load("pickle/test_info.gz")
        train_info = joblib.load("pickle/train_info.gz")
        protein_coding = joblib.load("pickle/protein_coding.gz")
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

        joblib.dump(test_info, "pickle/test_info.gz", compress="lz4")
        joblib.dump(train_info, "pickle/train_info.gz", compress="lz4")
        joblib.dump(protein_coding, "pickle/protein_coding.gz", compress="lz4")

    for sp in species:
        if Path(f"pickle/{sp}_regions.gz").is_file():
            continue
        print(sp)
        genome, ga = cm.parse_genome(f"data/species/{sp}/genome.fa", bin_size)
        regions = pd.read_csv(f"data/species/{sp}/windows.bed", sep="\t",
                                    index_col=False, names=["chrom", "start", "end"])
        regions["mid"] = (regions["start"] + (regions["end"] - regions["start"]) / 2)
        chromosomes = regions['chrom'].unique().tolist()
        chromosomes = [chrom for chrom in chromosomes if re.match("chr([0-9]*|X)$", chrom)]
        regions = regions.astype({"mid": int})
        regions = regions[regions['chrom'].isin(chromosomes)]
        regions = regions[['chrom', 'mid']].values.tolist()

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

        joblib.dump(one_hot, f"pickle/{sp}_one_hot.gz", compress="lz4")
        joblib.dump(ga, f"pickle/{sp}_ga.gz", compress=3)
        joblib.dump(regions, f"pickle/{sp}_regions.gz", compress="lz4")
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
