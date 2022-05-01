import os
import re
from pathlib import Path
from main_params import MainParams


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def find_between_r(s, first, last):
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


p = MainParams()
Path("controls").mkdir(parents=True, exist_ok=True)
for i in range(len(p.species)):
    tracks_folder = p.tracks_folder + p.species[i] + "/"
    track_names = []
    control = []
    for filename in os.listdir(tracks_folder):
        if filename.endswith(".gz"):
            track = filename[:-len(".100nt.bed.gz")]
            split = track.split(".")
            desc = track.replace(".", " ").replace("_", " ")
            display = split[3]
            seq_protocol = split[0]
            val_type = split[2]
            sample_id = split[4]
            meta = f"format=bed;sampleID={sample_id};seq_protocol={seq_protocol};bioProject=dl_gene_expression;genome={p.species[i]};val_type={val_type}"
            control.append(f"/osc-fs_home/ramzan/data/zenbu/tracks/{p.species[i]}/{filename}\t{display}\t{desc}\t{meta}")

    with open(f"controls/{p.species[i]}_control.tsv", "w+") as myfile:
        myfile.write("\n".join(control))
