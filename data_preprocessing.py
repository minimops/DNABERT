import random
import numpy.random
import pandas as pd
from math import floor
from Bio import SeqIO
from statistics import mean, median
import os
from DNABERT.motif.motif_utils import seq2kmer
from DNABERT.run_funs import create_dir, create_data_info_file


# converts a directory of fasta files into a list of strings
def parse_fasta(dir_: str, lol=False):
    directory = os.fsencode(dir_)
    seqList = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".fasta"):
            if lol:
                l = []
                for seq_record in SeqIO.parse(os.path.join(dir_, filename), "fasta"):
                    l.append(str(seq_record.seq))
                seqList.append(l)
            else:
                for seq_record in SeqIO.parse(os.path.join(dir_, filename), "fasta"):
                    seqList.append(str(seq_record.seq))
    return seqList


# for each string:
# split or sample into sequences <510
# generate random numbers until sum is greater
# TODO this is suuper dirty but was rushed for now, getting into python
# non overlapping splitting
def split_sequences_no(seq_list, low_b=5, upp_b=510, rat_max=.5):
    if (low_b >= upp_b):
        raise ValueError('upp_b has to be bigger than low_b')
    split_seq_list = []
    for seq in seq_list:
        n = []
        diff = len(seq)
        while diff > 0:
            # bias sample if full length
            n.append(min(sample_cut_length(low_b, upp_b, rat_max), diff))
            diff = len(seq) - sum(n)

        start = 0
        d = []
        for i in n:
            d.append(seq[start:start + i])
            start += i
        split_seq_list.extend(d)
    return split_seq_list


# random sampling splitting
# TODO
#  for now this takes the expected amt of samples as direct splitting creates as a guide for the amount of samples
#  to create per sequence (so individually adjusted for each seq length)
def split_sequences_rand(seq_list, low_b=5, upp_b=510, rat_max=.5):
    split_seq_list = []

    # expected average seq length
    expL = ((((upp_b - low_b) / 2) * (1 - rat_max)) + upp_b * rat_max)
    # for each sequence
    for seq in seq_list:
        # TODO maybe not floor
        amt = floor(len(seq) / expL)
        for i in range(amt):
            cutLength = sample_cut_length(low_b, min(upp_b, len(seq)), rat_max)
            # double plus one cuz of range and to include possibility of up to end
            cutStart = numpy.random.randint(0, len(seq) - cutLength + 1 + 1)
            split_seq_list.append(seq[cutStart:cutStart + cutLength])

    return split_seq_list


# samples a single cut length
def sample_cut_length(low_b, upp_b, rat_max):
    biasInd = numpy.random.choice(numpy.array([True, False], dtype=bool),
                                  p=[rat_max, 1 - rat_max])
    if biasInd:
        return upp_b
    else:
        return numpy.random.randint(low_b, upp_b + 1)


# load into seq to kmer function
def list2kmer(spl_sequences, k):
    fin = []
    for spl in spl_sequences:
        fin.append(seq2kmer(spl, k))
    return fin


# remove genomes with 'N' entries
def removeNAseq(seqList):
    for seq in seqList:
        if "N" in seq:
            seqList.remove(seq)
    return seqList


# remove genomes with 'N' entries
def removeNAseq_ft(seqList):
    for seq in seqList:
        if type(seq) is list:
            for s in seq:
                if "N" in s:
                    seq.remove(s)
            if len(seq) == 0:
                seqList.remove(seq)
        else:
            if "N" in seq:
                seqList.remove(seq)
    return seqList


# returns 'amt' number of random sequences of a list
def sampleSeqs(seqList, amt: int):
    # if amt not in range(0, len(seqList)):
    #     raise ValueError('amt needs to be smaller than the input length')
    random.shuffle(seqList)
    return seqList[:amt]


def pt_data_process(dirs_list, name, path, kmer, add_info='', low_b=5, upp_b=510, rat_max=.5):
    location = create_dir(name, path)
    lines = []
    split_seqs_both = []
    # TODO think this is really inefficient cuz of a lot of moving memory, fix could be simple
    for d in dirs_list:
        seqs = removeNAseq(parse_fasta(d))
        n_o_cut = split_sequences_no(seqs, low_b, upp_b, rat_max)
        samp_cut = split_sequences_rand(seqs, low_b, upp_b, rat_max)
        split_seqs_both.extend(n_o_cut)
        split_seqs_both.extend(samp_cut)

        lines.extend(["\n",
                      "Data from dir: " + d,
                      "Number of non NA sequences: " + str(len(seqs)),
                      "Median length of seq: " + str(median(map(len, seqs))),
                      "Average length of seq: " + str(mean(map(len, seqs))),
                      "Split into sequences of length: " + str(low_b) + " - " + str(upp_b),
                      "With bias probability to max len of: " + str(rat_max),
                      "Number of non-overlap sub-seqs: " + str(len(n_o_cut)),
                      "Number of sampled sub-seqs: " + str(len(samp_cut)),
                      "Kmer: " + str(kmer)
                      ])

    lines.append(add_info)

    random.shuffle(split_seqs_both)

    # write as interim text file
    textfile = open(location + '/split_interim.txt', "w")
    for element in split_seqs_both:
        textfile.write(element + "\n")
    textfile.close()

    # create kmers
    kmer_seqs_all = list2kmer(split_seqs_both, kmer)

    # write final kmers as text file
    textfile = open(location + '/full_kmers.txt', "w")
    for element in kmer_seqs_all:
        textfile.write(element + "\n")
    textfile.close()

    # write info file
    create_data_info_file(location, lines)


def ft_split_sequences_rand(seq_list, sample_amt, cutLength):
    split_seq_list = []
    for seq in seq_list:
        if len(seq) <= cutLength:
            print("left out a seq because of seq length of: " + str(len(seq)))
            continue
        for i in range(sample_amt):
            # double plus one cuz of range and to include possibility of up to end
            cutStart = numpy.random.randint(0, len(seq) - cutLength + 1 + 1)
            split_seq_list.append(seq[cutStart:cutStart + cutLength])
    return split_seq_list


# idea
# there is a maximum cap per fasta file of 256
# calc average and median of seq len per fasta file
# half of this is non overlap splitting
# get all starts
# if greater than half of cap, sample half cap amount
# other half is randomly sampled starts
# randomly sample half cap amount

def ft_df_creation_2(class_dirs, cap, split, cutlength, kmer, labels=0):
    n_o_number = int(cap * split)
    sam_number = cap - n_o_number

    cl_df_list = []
    label_iter = 0

    for c in class_dirs:
        split_seq_list = []
        list_seqs = parse_fasta(c, lol=True)
        list_seqs = removeNAseq_ft(list_seqs)

        # for every fasta file:
        for los in list_seqs:
            n_o_list = []
            sam_list = []
            # for every sequence in file
            for seq in los:
                if len(seq) <= cutlength:
                    print("left out a seq because of seq length of: " + str(len(seq)))
                    continue
                # create seq from 1 to end by cutlength
                starts_no = list(range(0, len(seq) + 1, cutlength))
                # cut out sequences and add to no list
                n_o_list.extend([seq[i:i+cutlength] for i in starts_no])
                # create random starts same number as n-o starts
                starts_sam = numpy.random.randint(0, len(seq) - cutlength + 1 + 1, len(starts_no))
                sam_list.extend([seq[i:i+cutlength] for i in starts_sam])

            # sample elements of lists to match cap
            if len(n_o_list) > n_o_number:
                n_o_list = random.sample(n_o_list, n_o_number)
            if len(sam_list) > sam_number:
                sam_list = random.sample(sam_list, sam_number)
            split_seq_list.extend(n_o_list)
            split_seq_list.extend(sam_list)
        # create kmers
        split_seq_kmer_list = list2kmer(split_seq_list, kmer)
        # create df
        cl_df_list.append(pd.DataFrame({'sequence': split_seq_kmer_list,
                                        'label': labels if labels != 0 else label_iter}))
        label_iter += 1
    # all classes together
    full_df = pd.concat(cl_df_list)
    # shuffle
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    return full_df


def ft_df_creation(class_dirs, sample_amt, kmer, cutLength, labels=0):
    cl_df_list = []
    label_iter = 0
    for c in class_dirs:
        split_seq_list = []  # placeholder
        list_seqs = parse_fasta(c, lol=True)
        list_seqs = removeNAseq_ft(list_seqs)
        # create list of lists of sequences
        # for every listentry do the following x times
        for los in list_seqs:
            iters = int(sample_amt / len(los))
            split_seq_list.extend(ft_split_sequences_rand(los, iters, cutLength))

        # create kmers
        split_seq_kmer_list = list2kmer(split_seq_list, kmer)
        cl_df_list.append(pd.DataFrame({'sequence': split_seq_kmer_list,
                                        'label': labels if labels != 0 else label_iter}))
        label_iter += 1

    full_df = pd.concat(cl_df_list)
    # shuffle
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    return full_df


def ft_data_process(dirlists, name, path, cap, split, cutLength, kmer, add_info='', labels=0):
    # TODO missing assert str and len of dirList
    # create dir
    location = create_dir(name, path)
    lines = []
    # for test and train dirlists
    filenameList = ['dev', 'train']
    for i in range(len(dirlists)):
        # write train/test file
        ft_df_creation_2(class_dirs=dirlists[i], cap=cap, split=split, cutlength=cutLength,
                         kmer=kmer, labels=labels).to_csv(
            location + "/" + filenameList[i] + ".tsv", sep='\t', index=False)
    # write info file
    lines.extend(["\n",
                  "Data from dirs: " + str(list(map(', '.join, dirlists))),
                  "Cut length: " + str(cutLength),
                  "Kmer: " + str(kmer)
                  ])
    lines.append(add_info)
    create_data_info_file(location, lines)
