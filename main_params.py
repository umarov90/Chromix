import os
import math
from pathlib import Path
import pathlib


class MainParams:
    def __init__(self):
        self.input_size = 420100
        self.half_size = int(self.input_size / 2)
        self.bin_size = 100
        self.hic_bin_size = 10000
        self.num_hic_bins = 40
        self.half_size_hic = 200000
        self.num_bins = 1001
        self.half_num_regions = int(self.num_bins / 2)
        self.mid_bin = math.floor(self.num_bins / 2)
        self.BATCH_SIZE = 1
        self.GLOBAL_BATCH_SIZE = 1
        self.STEPS_PER_EPOCH = 5
        self.num_epochs = 1000
        self.hic_track_size = 1
        self.num_features = 4
        self.shift_speed = 2000000
        self.initial_shift = 250 # +- 50k
        self.hic_size = 780
        self.species = ["hg38", "mm10", "macFas5", "calJac4", "rheMac8", "canFam3", "oviAri4", "rn6"]
        script_folder = pathlib.Path(__file__).parent.resolve()
        folders = open(str(script_folder) + "/data_dirs").read().strip().split("\n")
        os.chdir(folders[0])
        self.parsed_tracks_folder = folders[1]
        self.parsed_hic_folder = folders[2]
        self.model_folder = folders[3]
        self.model_name = "hic_model.h5"
        self.model_path = self.model_folder + self.model_name
        self.figures_folder = "figures_1"
        self.tracks_folder = "/Users/ramzan/variants_100/tracks/"
        self.temp_folder = "/home/user/data/temp/"
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "plots").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)