import os
import math
import pandas as pd
from pathlib import Path
import pathlib
import joblib

class MainParams:
    def __init__(self):
        self.input_size = 651 * 3 * 128
        self.half_size = self.input_size // 2
        self.bin_size = 128
        self.num_all_bins = self.input_size // self.bin_size
        self.hic_bin_size = 2000
        self.hic_size = 4851
        self.num_hic_bins = 100
        self.half_size_hic = (self.num_hic_bins // 2) * self.hic_bin_size
        self.num_bins = 651 # 783 # 1563
        self.half_num_regions = self.num_bins // 2
        self.mid_bin = self.num_bins // 2
        self.BATCH_SIZE = 1
        self.NUM_GPU = 4
        self.GLOBAL_BATCH_SIZE = self.NUM_GPU * self.BATCH_SIZE
        self.predict_batch_size = self.GLOBAL_BATCH_SIZE
        self.w_step = 90
        self.STEPS_PER_EPOCH = 200
        self.num_epochs = 1000
        self.num_features = 4
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
        self.tracks_folder_sc = folders[7]
        self.species_folder = folders[8]
        self.model_name = "chromix_" + str(self.input_size)
        self.model_path = self.model_folder + self.model_name
        self.figures_folder = "figures_1"
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "plots").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)
        self.species = ["hg38", "mm10"]
        self.hic_keys = {}
        for specie in ["hg38", "mm10"]:
            self.hic_keys[specie] = pd.read_csv(f"data/{specie}_hic.tsv", sep="\t", header=None).iloc[:, 0]
        self.loss_weights = {"hg38_expression": 1.0, "hg38_epigenome": 1.0, "hg38_conservation": 2.0, "hg38_hic": 1.0, "hg38_expression_sc": 8.0,
                    "mm10_expression": 1.0, "mm10_epigenome": 1.0, "mm10_hic": 1.0}
        self.lrs = {"stem": 0.0001, "body": 0.0001, "3d_projection": 0.0001,
                    "expression": 0.0001, "epigenome":  0.0001, "hg38_conservation":  0.0001, "hic":  0.0001, "hg38_expression_sc": 0.0001}
        self.wds = {"stem": 0.0, "body": 0.0001, "3d_projection": 0.0,
                    "expression": 0.0, "epigenome": 0.0, "hg38_conservation": 0.0, "hic": 0.0, "hg38_expression_sc": 0.0}
        self.tss_only = False
