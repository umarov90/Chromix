import os
import re
import pathlib
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import common as cm
import matplotlib
import model as mo
import main_params
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

p = main_params.MainParams()
script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])
model_folder = folders[3]
heads = joblib.load("pickle/heads.gz")
head_id = 0
head_tracks = heads[head_id]
one_hot = joblib.load("pickle/one_hot.gz")

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = tf.keras.models.load_model(model_folder + p.model_name,
                                           custom_objects={'PatchEncoder': mo.PatchEncoder})
    our_model.get_layer("our_head").set_weights(joblib.load(model_folder + p.model_name + "_head_" + str(head_id)))


def calculate_score(chrn, pos, ref, alt, element):
        ind_ref = cm.nuc_to_ind(ref)
        ind_alt = cm.nuc_to_ind(alt)
        gene = genes.loc[genes['gene_name'] == element]["start"].values[0]
        start = gene - half_size
        seq = one_hot[chrn][gene - half_size, gene + half_size + 1]
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
