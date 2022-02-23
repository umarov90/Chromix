import os
import re
import pathlib
import joblib
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import stats
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
p = main_params.MainParams()
script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])
model_folder = folders[3]
heads = joblib.load("pickle/heads.gz")
head_id = 0
head_tracks = heads[head_id]
one_hot = joblib.load("pickle/one_hot.gz")

genes = pd.read_csv("data/gencode.v39.annotation.gtf.gz",
                  sep="\t", comment='#', names=["chr", "h", "type", "start", "end", "m1", "strand", "m2", "info"],
                  header=None, index_col=False)
genes = genes[genes.type == "gene"]
genes["gene_name"] = genes["info"].apply(lambda x: re.search('gene_name "(.*)"; level', x).group(1)).copy()
genes.drop(genes.columns.difference(['chr', 'start', "end", "gene_name"]), 1, inplace=True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = tf.keras.models.load_model(model_folder + p.model_name)
    our_model.get_layer("our_head").set_weights(joblib.load(model_folder + p.model_name + "_head_" + str(head_id)))


df = pd.read_csv("data/GRCh38_ALL.tsv", sep="\t")
# one_hot["chr2"][60494941-1]
seqs1 = []
seqs2 = []
for index, row in df.iterrows():
    if index % 1000 == 0:
        print(index, end=" ")
    ind_ref = cm.nuc_to_ind(row['Ref'])
    ind_alt = cm.nuc_to_ind(row['Alt'])
    chrom = "chr" + str(row['Chromosome'])
    sbt = one_hot[chrom][row['Position'] - 1][ind_ref]
    # gene = genes.loc[genes['gene_name'] == row['Element']]["start"].values[0] - 1
    start = row['Position'] - p.half_size - 1

    correction = 0
    if start < 0:
        correction = start
        start = 0
    if start + p.input_size > len(one_hot[chrom]):
        extra = (start + p.input_size) - len(one_hot[chrom])
        start = start - extra
        correction = extra

    snp_pos = p.half_size + correction

    seq = one_hot[chrom][start: start + p.input_size + 1].copy()

    seqs1.append(seq[:-1].copy())
    if ind_alt != -1:
        if seq[snp_pos][ind_ref] != 1:
            print("Problem")
        seq[snp_pos][ind_ref] = 0
        seq[snp_pos][ind_alt] = 1
        seqs2.append(seq[:-1].copy())
    else:
        seqs2.append(np.delete(seq, snp_pos, axis=0))

NUM_POINTS = 5000
gt_effect = np.squeeze(df['Value'].to_numpy())[:NUM_POINTS]
print(gt_effect.shape)
print(f"Predicting {len(seqs1)}")
effect = mo.batch_predict_effect(our_model, seqs1[:NUM_POINTS], seqs2[:NUM_POINTS])
print("Done")
# joblib.dump(effect, "temp/effect.p")
# effect = joblib.load("temp/effect.p")
# effect = np.mean(effect, axis=-1)

X_train, X_test, Y_train, Y_test = train_test_split(effect, gt_effect, test_size=0.4, random_state=1)
clf = RandomForestRegressor(random_state=0, max_depth=500) # max_depth=100,
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

corr = stats.pearsonr(Y_test, Y_pred)[0]
print("Correlation: " + str(corr))
