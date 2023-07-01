import os
import math
from pathlib import Path
import pathlib

import joblib
import pandas as pd


class MainParams:
    def __init__(self):
        self.input_size = 2349 * 128# 256128
        self.tss_only = False
        self.dim = 1024 * 2
        self.lr = 4e-04
        self.half_size = self.input_size // 2
        self.bin_size = 128
        self.num_all_bins = self.input_size // self.bin_size
        self.hic_bin_size = 2000
        self.hic_size = 4851
        self.num_hic_bins = 100
        self.half_size_hic = (self.num_hic_bins // 2) * self.hic_bin_size
        self.num_bins = 783 # 1001
        self.half_num_regions = self.num_bins // 2
        self.mid_bin = self.num_bins // 2
        self.accum_iter = 1
        self.BATCH_SIZE = 4
        self.pred_batch_size = 2 * self.BATCH_SIZE
        self.w_step = 50 * self.BATCH_SIZE
        self.MAX_STEPS_PER_EPOCH = 200
        self.num_epochs = 10000000
        self.num_features = 4
        self.species = ["hg38", "mm10"]
        self.script_folder = pathlib.Path(__file__).parent.resolve()
        folders = open(str(self.script_folder) + "/../data_dirs").read().strip().split("\n")
        self.data_folder = folders[0]
        os.chdir(self.data_folder)
        self.parsed_tracks_folder = folders[1]
        self.hic_folder = folders[2]
        self.model_folder = folders[3]
        self.temp_folder = folders[4]
        self.pickle_folder = folders[5]
        self.tracks_folder = folders[6]
        self.species_folder = folders[7]
        # self.tracks_folder_sc = folders[7]
        self.model_name = "chromix_" + str(self.input_size)
        self.model_path = self.model_folder + self.model_name
        self.figures_folder = "figures_1"
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "plots").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)
        self.loss_weights = {"expression": 40.0, "epigenome": 10.0, "conservation": 1.0, "hic": 8.0, "expression_sc": 1000}
        self.output_heads = {}
        if Path(f"{self.pickle_folder}heads.gz").is_file():
            self.heads = joblib.load(f"{self.pickle_folder}heads.gz")
        for head_key in self.heads.keys():
            if isinstance(self.heads[head_key], dict):
                for key2 in self.heads[head_key].keys():
                    print(f"Number of tracks in head {head_key} {key2}: {len(self.heads[head_key][key2])}")
                    self.output_heads[head_key + "_" + key2] = len(self.heads[head_key][key2])
            else:
                print(f"Number of tracks in head {head_key}: {len(self.heads[head_key])}")
                self.output_heads[head_key + "_expression"] = len(self.heads[head_key])
        self.hic_keys = {}
        for specie in ["hg38", "mm10"]:
            self.hic_keys[specie] = pd.read_csv(f"data/{specie}_hic.tsv", sep="\t", header=None).iloc[:, 0]
        self.need_load = True
        self.running_loss = {}
        for specie in ["hg38", "mm10"]:
            self.running_loss[specie] = {}
        self.running_loss["total"] = []
