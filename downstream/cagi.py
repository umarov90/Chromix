import os
import re

import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import gc
import random
import pandas as pd
import math
import numpy as np
import common as cm
from pathlib import Path
import pickle
import matplotlib
import model as mo
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
matplotlib.use("agg")


# calculate multiple effects with slight shifts to the start
os.chdir(open("data_dir").read().strip())
input_size = 100000
half_size = input_size / 2
model_path = "model1/expression_model_1.h5"
bin_size = 1000
num_regions = int(input_size / bin_size)
mid_bin = math.floor(num_regions / 2)

genes = pd.read_csv("gencode.v38.annotation.gtf.gz",
                  sep="\t", comment='#', names=["chr", "h", "type", "start", "end", "m1", "strand", "m2", "info"],
                  header=None, index_col=False)
genes = genes[genes.type == "gene"]
genes["gene_name"] = genes["info"].apply(lambda x: re.search('gene_name "(.*)"; level', x).group(1)).copy()
genes.drop(genes.columns.difference(['chr', 'start', "end", "gene_name"]), 1, inplace=True)

if Path("pickle/genome.p").is_file():
    genome = pickle.load(open("pickle/genome.p", "rb"))
else:
    genome, ga = cm.parse_genome("hg38.fa", 1000)
    pickle.dump(genome, open("pickle/genome.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

our_model = tf.keras.models.load_model(model_path, custom_objects={'PatchEncoder': mo.PatchEncoder})


def calculate_score(chrn, pos, ref, alt, element):
        ind_ref = cm.nuc_to_ind(ref)
        ind_alt = cm.nuc_to_ind(alt)
        gene = genes.loc[genes['gene_name'] == element]["start"].values[0]
        start = gene - half_size
        seq = genome[chrn][gene - half_size, gene + half_size + 1]
        pos = pos - start
        a1 = our_model.predict(seq[:-1])
        if ind_alt != -1:
            if seq[pos][ind_ref] != 1:
                print("Problem")
            seq[pos][ind_ref] = 0
            seq[pos][ind_alt] = 1
            a2 = our_model.predict(seq)
        else:
            a2 = our_model.predict(np.delete(seq, pos))
        effect = a1[mid_bin] - a2[mid_bin]
        return effect


df = pd.read_csv("GRCh38_ALL.tsv", sep="\t")
df['our_score'] = df.apply(lambda row: calculate_score(row['Chromosome'], row['Position'],
                                                       row['Ref'], row['Alt'], row['Element']), axis=1)
corr = df['our_score'].corr(df['Value'])
print("Correlation: " + str(corr))
