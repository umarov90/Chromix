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
from liftover import get_lifter
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

converter = get_lifter('hg19', 'hg38')
p = MainParams()
head = joblib.load(f"{p.pickle_folder}heads.gz")
head_tracks = head["sc"]
one_hot = joblib.load(f"{p.pickle_folder}one_hot.gz")

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, head_tracks)
    # since model created with one head, that heads name will be our_expression
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_sc"))

# all_traits.matched_ancestry.snp_with_proxy.bed.gz
df = pd.read_csv("data/EFO_0000729_ulcerative_colitis.matched_ancestry.snp_with_proxy.bed.gz", sep="\t", index_col=False,
                      names=["chrom", "start", "end", "info", "score", "strand"], comment="#")
df = df.sample(frac=1).reset_index(drop=True)
disease_names = []
effects = []
track_norm = []
for track in head_tracks:
    effects.append([])
    track_norm.append([])


for index, row in df.iterrows():
    if index % 100 == 0:
        print(index, end=" ")
    try:
        disease_name = row["info"].split("|")[6]
        if disease_name not in disease_names:
            disease_names.append(disease_name)
            for track_to_use, track in enumerate(head_tracks):
                effects[track_to_use].append([])
        disease_index = disease_names.index(disease_name)

        seqs = []
        start = converter[row["chrom"]][row["start"]][0][1] - p.half_size # 1 based?
        if row["chrom"] not in one_hot.keys():
            continue
        if start < 0 or start + p.input_size > len(one_hot[row["chrom"]]):
            continue
        seq1 = one_hot[row["chrom"]][start: start + p.input_size][..., :-1]
        seqs.append(seq1)
        seq2 = seq1.copy()
        seq2[p.half_size - 100: p.half_size + 100] = 0
        seqs.append(seq2)
        vals = our_model.predict(mo.wrap2(np.asarray(seqs), p.predict_batch_size))

        for track_to_use, track in enumerate(head_tracks):
            # if track.startswith("scATAC"):
            #     continue
            effect = np.max(np.absolute((vals[0][track_to_use, :] - vals[1][track_to_use, :])))
            effects[track_to_use][disease_index].append(effect)
            track_norm[track_to_use].append(np.sum(vals[0][track_to_use, :]).clip(min=0.00001))
        # ranked = [x for _, x in sorted(zip(maes, head_tracks), reverse=True)]
        # with open(f"ranked/{disease_name}.csv", "w+") as myfile:
        #     myfile.write("\n".join(ranked))
    except Exception:
        pass
    if index > 50000:
        break

joblib.dump(effects, "effects.p")
joblib.dump(disease_names, "disease_names.p")
joblib.dump(track_norm, "track_norm.p")
# effects = joblib.load("effects.p")
# disease_names = joblib.load("disease_names.p")
# track_norm = joblib.load("track_norm.p")

# np.savetxt("track0.csv", effects[0][0])
# np.savetxt("tracknorm0.csv", track_norm[0])

for track_to_use, track in enumerate(head_tracks):
    # if track.startswith("scATAC"):
    #     continue
    for disease_index, disease_name in enumerate(disease_names):
        effects[track_to_use][disease_index] = np.percentile(effects[track_to_use][disease_index], 99) / np.percentile(track_norm[track_to_use], 99)

effects = np.asarray(effects)

short_names = []
for track in head_tracks:
    short_names.append(track[:track.rfind(".cluster")])

effects = pd.DataFrame(data=np.squeeze(effects), index=short_names)
# effects = effects[effects.columns.drop(list(effects.filter(regex='scATAC')))]
effects = effects[~effects.index.str.contains("scATAC")]
effects.to_csv("effects.tsv", sep="\t", header=False)


effects = pd.DataFrame(data=effects.T, index=disease_names, columns=short_names)
effects = effects[effects.columns.drop(list(effects.filter(regex='scATAC')))]

g = sns.heatmap(effects)
# g = sns.clustermap(effects, rasterized=True, figsize=(16, 4), yticklabels=True, xticklabels=True)
plt.savefig(f"colitis.svg")
print("Done")
