import pathlib

import pandas as pd
import os
import re

script_folder = pathlib.Path(__file__).parent.resolve()
folders = open(str(script_folder) + "/../data_dirs").read().strip().split("\n")
os.chdir(folders[0])

genes = pd.read_csv("data/gencode.v39.annotation.gtf.gz",
                    sep="\t", comment='#', names=["chr", "h", "type", "start", "end", "m1", "strand", "m2", "info"],
                    header=None, index_col=False)
genes = genes[genes.type == "gene"]
genes["gene_name"] = genes["info"].apply(lambda x: re.search('gene_name "(.*)"; level', x).group(1)).copy()
genes.drop(genes.columns.difference(['chr', 'start', "end", "gene_name"]), 1, inplace=True)

genes.to_csv("gencode.v39.bed", sep="\t", index=False, header=False)
# ./bedtools intersect -a enformer_test.bed -b gencode.v39.bed > enformer_gencode.bed