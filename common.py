import numpy as np
import math
import re
from numba import jit


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


enc_mat = np.append(np.eye(4),
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], axis=0)
enc_mat = enc_mat.astype(bool)
mapping_pos = dict(zip("ACGTRYSWKMBDHVN", range(15)))


def encode(fa, pos, seq_len):
    half_size = int((seq_len - 1) / 2)
    if (pos - half_size < 0):
        enc_seq = "N" * (half_size - pos) + fa[0: pos + half_size + 1]
    elif (pos + half_size + 1 > len(fa)):
        enc_seq = fa[pos - half_size: len(fa)] + "N" * (half_size + 1 - (len(fa) - pos))
    else:
        enc_seq = fa[pos - half_size: pos + half_size + 1]

    return encode_seq(enc_seq)


def encode_seq(seq):
    try:
        seq2 = [mapping_pos[i] for i in seq]
        return enc_mat[seq2]
    except:
        print(seq)
        return None


def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


def parse_genome(g, bin, chr1=False):
    fasta = {}
    ga = {}
    seq = ""
    with open(g) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0 and re.match("chr([0-9]*|X)$", chrn):
                    seq = clean_seq(seq)
                    fasta[chrn] = seq
                    ga[chrn] = np.zeros(int(len(seq) / bin) + 1, dtype=np.float32)
                    print(chrn + " - " + str(len(seq)))
                    if chr1:
                        return fasta
                chrn = line.strip()[1:]
                try:
                    chrn = line.strip()[1:]
                except Exception as e:
                    pass
                seq = ""
            else:
                seq += line
        if len(seq) != 0 and re.match("chr([0-9]*|X)$", chrn):
            seq = clean_seq(seq)
            fasta[chrn] = seq
            ga[chrn] = np.zeros(int(len(seq) / bin) + 1, dtype=np.float32)
            print(chrn + " - " + str(len(seq)))
    return fasta, ga


def parse_bed(reg_elements, path):
    with open(path) as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            chrp = int(vals[7]) - 1
            reg_elements.setdefault(chrn, []).append(chrp)


def change_seq(x):
    return rev_comp(x)
    
    
def rev_comp(s):
    reversed_arr = s[::-1]
    vals = []
    for v in reversed_arr:
        if v[0]:
            vals.append([False, False, False, True, v[4]])
        elif v[1]:
            vals.append([False, False, True, False, v[4]])
        elif v[2]:
            vals.append([False, True, False, False, v[4]])
        elif v[3]:
            vals.append([True, False, False, False, v[4]])
        else:
            vals.append([False, False, False, False, v[4]])
    return np.array(vals, dtype=bool)


def nuc_to_ind(nuc):
    nuc = nuc.upper()
    ind = -1
    if nuc == "A":
        ind = 0
    elif nuc == "C":
        ind = 1
    elif nuc == "G":
        ind = 2
    elif nuc == "T":
        ind = 3
    return ind

@jit(nopython=True)
def to_dna(a):
    seq = ""
    for v in a:
        if v[0]:
            seq+="A"
        elif v[1]:
            seq+="C"
        elif v[2]:
            seq+="G"
        elif v[3]:
            seq+="T"
        else:
            seq+="N"
    return seq

def get_human_readable(size, precision=2):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1  # increment the index of the suffix
        size = size / 1024.0  # apply the division
    return "%.*f%s" % (precision, size, suffixes[suffixIndex])


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def store(param, alist):
    alist.append(param)
    return param


def change_seq(x):
    return rev_comp(x)


def find_overlapping_tuples(tuples_list, bedgraph_file):
    overlapping_tuples = []
    with open(bedgraph_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            for tup in tuples_list:
                tup_chrom = tup[0]
                tup_pos = tup[1]

                if chrom == tup_chrom and start <= tup_pos <= end:
                    overlapping_tuples.append(tup)

    return list(set(overlapping_tuples))