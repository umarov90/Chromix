import os
import pathlib
import model as mo
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import attribution
from sklearn.cluster import KMeans
import seaborn as sns
from main_params import MainParams
import common as cm
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

DISEASE_NUM = 400
# /osc-fs_home/hon-chun/analysis/tenX_single_cell/paper_run/GWAS_enrichment/func_map/LDBlockGenerator/immuno.n12.hg19.p_5e-08.r2_0.2.kb_1000/bed/00_all_traits.matched_ancestry.snp_with_proxy.bed.gz
# /osc-fs_home/hon-chun/analysis/tenX_single_cell/GWAS_enrichment/gchromVar/test_run/00_selected.trait.n122.txt
# /osc-fs_home/hon-chun/analysis/tenX_single_cell/GWAS_enrichment/gchromVar/test_run/input/snp/all_credible/
# /osc-fs_home/hon-chun/analysis/tenX_single_cell/GWAS_enrichment/gchromVar/test_run/selected_traits_n122

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

p = MainParams()
head = joblib.load(f"{p.pickle_folder}heads.gz")
head_tracks = head["sc"]
one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")

model_path = p.model_folder + "0.8025195642453672_0.4928313747161716/" + p.model_name
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head_tracks)
    # since model created with one head, that heads name will be our_expression
    our_model.get_layer("our_expression").set_weights(joblib.load(model_path + "_sc"))


df = pd.read_csv("data/all_traits.matched_ancestry.snp_with_proxy.bed", sep="\t", index_col=False,
                      names=["chrom", "start", "end", "info", "score", "strand"], comment="#")

disease_names = []
effects = []
for track in head_tracks:
    effects.append([])

for index, row in df.iterrows():
    disease_name = row["info"].split("|")[6]
    if disease_name not in disease_names:
        disease_names.append(disease_name)
        for track_to_use, track in enumerate(head_tracks):
            effects[track_to_use].append([])
    disease_index = disease_names.index("disease_name")

    seqs1 = []
    seqs2 = []
    
    start = row["start"] - p.half_size - 1 # 1 based, right?
    if row["chrom"] not in one_hot.keys():
        continue
    if start < 0 or start + p.input_size > len(one_hot[row["chrom"]]):
        continue
    seq1 = one_hot[row["chrom"]][start: start + p.input_size]
    # one_hot[row["chrom"]][row["start"] - 1: row["start"]]
    seqs1.append(seq1)
    for j in range(4):
        if seq1[p.half_size][j]:
            continue
        seq2 = seq1.copy()
        seq2[p.half_size] = [False, False, False, False]
        seq2[p.half_size][j] = True
        seqs2.append(seq2)
    vals1 = our_model.predict(mo.wrap2(np.asarray(seqs1), 16))
    vals2 = our_model.predict(mo.wrap2(np.asarray(seqs2), 16))

    vals1 = np.sum(vals1, axis=-1)
    vals2 = np.sum(vals2, axis=-1)

    for track_to_use, track in enumerate(head_tracks):
        # mae = np.sum(np.absolute((vals1[:, track_to_use, :] - vals2[:, track_to_use, :])))
        mae = 0
        for a in range(len(seqs1)):
            mean1 = np.mean(np.absolute((vals1[a, :] - vals2[3 * a, :])))
            mae1 = np.absolute((vals1[a, track_to_use] - vals2[3 * a, track_to_use]))
            mae1 = mae1 / mean1 - 1
            mean2 = np.mean(np.absolute((vals1[a, :] - vals2[3 * a + 1, :])))
            mae2 = np.absolute((vals1[a, track_to_use] - vals2[3 * a + 1, track_to_use]))
            mae2 = mae2 / mean2 - 1
            mean3 = np.mean(np.absolute((vals1[a, :] - vals2[3 * a + 2, :])))
            mae3 = np.absolute((vals1[a, track_to_use] - vals2[3 * a + 2, track_to_use]))
            mae3 = mae3 / mean3 - 1
            mae += max(mae1, mae2, mae3)
        effects[track_to_use][disease_index].append(mae)
    # ranked = [x for _, x in sorted(zip(maes, head_tracks), reverse=True)]
    # with open(f"ranked/{disease_name}.csv", "w+") as myfile:
    #     myfile.write("\n".join(ranked))
print("Taking mean")
for track_to_use, track in enumerate(head_tracks):
    for disease_index, disease_name in enumerate(disease_names):
        effects[track_to_use][disease_index] = np.mean(effects[track_to_use][disease_index])

effects = np.asarray(effects)
joblib.dump(effects, "temp/heat.p")

effects = joblib.load("temp/heat.p")
effects = pd.DataFrame(data=effects, index=disease_names, columns=track_names)
g = sns.clustermap(effects, rasterized=True, figsize=(16, 5*16), yticklabels=True, xticklabels=True)
plt.savefig(f"temp/clustermap_fold.svg")
