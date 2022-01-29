import math
from pathlib import Path


class MainParams:
    def __init__(self):
        self.input_size = 60001  # 210001 # 50001
        self.half_size = int(self.input_size / 2)
        self.bin_size = 200
        self.hic_bin_size = 10000
        self.num_hic_bins = 20
        self.half_size_hic = 100000
        self.num_regions = 201  # 501  # 201
        self.half_num_regions = int(self.num_regions / 2)
        self.mid_bin = math.floor(self.num_regions / 2)
        self.BATCH_SIZE = 4  # 1
        self.GLOBAL_BATCH_SIZE = 4 * self.BATCH_SIZE
        self.STEPS_PER_EPOCH = 300
        self.num_epochs = 2000
        self.hic_track_size = 1
        self.out_stack_num = 5000 #11529
        self.num_features = 5
        self.shift_speed = 2000000
        self.initial_shift = 300
        self.hic_size = 190
        self.model_folder = "/home/user/data/models/"
        self.model_name = "small.h5"
        self.model_path = self.model_folder + self.model_name
        self.figures_folder = "figures_1"
        self.tracks_folder = "/media/user/passport1/variants_100/tracks/"
        self.temp_folder = "/home/user/data/temp/"
        self.parsed_blocks_folder = "/media/user/30D4BACAD4BA9218/parsed_blocks/"
        self.parsed_tracks_folder = "/home/user/data/parsed_tracks/"
        # parsed_tracks_folder = "/media/user/30D4BACAD4BA9218/parsed_tracks/"
        self.chromosomes = ["chrX"]  # "chrY"
        for i in range(1, 23):
            self.chromosomes.append("chr" + str(i))

        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "plots").mkdir(parents=True, exist_ok=True)
        Path(self.figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)