import os
import pandas as pd
import sys
from subprocess import Popen
import re


def run_proc(command):
    p = Popen(command, shell=True)
    p.wait()


hisat2 = "/osc-fs_home/ramzan/software/hisat2-2.2.1/hisat2"
bam_to_ctss_bin = "/osc-fs_home/ramzan/software/bamToCTSS/v0.7/bamToCTSS_v0.7.pl"

meta = pd.read_csv(sys.argv[1], sep="\t")
wd = os.path.dirname(os.path.abspath(sys.argv[1])) + "/.."

for index, row in meta.iterrows():
    files = row["fastq_ftp"].split(";")
    name = row["sample_title"]
    end = name.find(" ")
    if end != -1:
        name = name[:end]
    name = re.sub('[^0-9a-zA-Z]+', '', name)
    if len(files) == 1:
        file_loc = "fastq" + files[0][files[0].rfind('/'):]
        # hisat2 mapping
        run_proc(
            f"{hisat2} -p 20 genome/index -U {file_loc} | samtools sort - -o {wd}/bam/{row['run_accession']}.bam",
        )
    else:
        file_loc1 = f"{wd}/fastq" + files[0][files[0].rfind('/'):]
        file_loc2 = f"{wd}/fastq" + files[1][files[1].rfind('/'):]
        # hisat2 mapping
        run_proc(
            f"{hisat2} -p 20 genome/index -1 {file_loc1} -2 {file_loc2} | samtools sort - -o {wd}/bam/{row['run_accession']}.bam",
        )

    # convert bam to ctss
    run_proc(
        f"perl {bam_to_ctss_bin} --bamPath={wd}/bam/{row['run_accession']}.bam --exclude_flag=512,256 --minLen=25 --maxLen=100 --min_MAPQ=3 --longShortFormat=short --outputPrefix={row['run_accession']} --outDir={wd}/bam_to_ctss/",
    )
    # make 100nt window signal
    run_proc(
        f"gzip -dc {wd}/bam_to_ctss/{row['run_accession']}/bed/{row['run_accession']}.short.ctss.bed.gz | cut -f 1,2,3,5 | sort -k1,1 -k2,2n | bedtools map -c 4 -o sum -a {wd}/genome/windows.100nt.bed -b stdin | awk '$4 != \".\" {{print}}' | gzip -c >{wd}/bed/CAGE.RNA.ctss.{row['run_accession']}.{name}.100nt.bed.gz;",
             )



