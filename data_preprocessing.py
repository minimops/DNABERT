import random
import numpy.random
import pandas as pd
from Bio import SeqIO
from statistics import mean, median
import os
from math import ceil
from motif.motif_utils import seq2kmer
from src.run_funs import create_dir, create_data_info_file


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
    if low_b >= upp_b:
        raise ValueError('upp_b has to be bigger than low_b')
    split_seq_list = []
    for seq in seq_list:
        if len(seq) < low_b:
            print("sequence skipped because of length < %s" % low_b)
            continue
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
        if len(seq) < low_b:
            print("sequence skipped because of length < %s" % low_b)
            continue
        # ceiling here to create at least some for short seqs
        amt = ceil(len(seq) / expL)
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

        lines.extend(["Data from dir: " + d,
                      "Number of non NA sequences: " + str(len(seqs)),
                      "Median length of seq: " + str(median(map(len, seqs))),
                      "Average length of seq: " + str(mean(map(len, seqs))),
                      "Split into sequences of length: " + str(low_b) + " - " + str(upp_b),
                      "With bias probability to max len of: " + str(rat_max),
                      "Number of non-overlap sub-seqs: " + str(len(n_o_cut)),
                      "Number of sampled sub-seqs: " + str(len(samp_cut)),
                      "Kmer: " + str(kmer)
                      ])

    lines.append("\n" + add_info)

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


def ft_df_creation(class_dirs, cap, cutlength, kmer, max_mult=1):
    cl_df_list = []
    label_iter = 0
    # just to count left out sequences
    lo_counter_list = []

    def cutting_fun(aseq, max_amt, min_amt=None, replace=False):
        last_possible_cut = len(aseq) - cutlength + 1
        if min_amt is None:
            min_amt = last_possible_cut
        starts_sam = numpy.random.choice(last_possible_cut, min(int(min_amt), int(max_amt)), replace=replace)
        return [aseq[i:i + cutlength] for i in starts_sam]

    for c in class_dirs:
        split_seq_list = []
        list_seqs = parse_fasta(c, lol=True)
        list_seqs = removeNAseq_ft(list_seqs)
        lo_counter = 0
        # for every fasta file:
        for los in list_seqs:
            sam_list = []
            rem_seq = []
            # for every sequence in file
            for seq in los:
                if len(seq) <= cutlength:
                    print("left out a seq because of seq length of: " + str(len(seq)))
                    rem_seq.append(seq)
                    lo_counter += 1
                    continue
                sam_list.extend(cutting_fun(seq, cap))

            # TODO this removing by value seems rather dumb here, not sure of a better fix rn
            # remove too short seqs
            for r in rem_seq:
                los.remove(r)

            # if we are underneath the cap
            # done this complex way to maximize diversity
            if len(sam_list) < cap and max_mult > 1:
                orig_sam_len = len(sam_list)
                if len(los) > 1:
                    cut_p_seq = (cap - len(sam_list)) / (len(los) * 5)
                    # while we are underneath cap or underneath max_mult
                    while len(sam_list) < cap:
                        if len(sam_list) / orig_sam_len > max_mult:
                            break
                        # choose seq at random
                        seq = los[numpy.random.randint(0, len(los))]
                        # cut x amt of subseqs at random and add to list
                        sam_list.extend(cutting_fun(seq, cut_p_seq))
                elif len(los) == 1:
                    seq = los[0]
                    # cut x amt of subseqs at random and add to list
                    sam_list.extend(cutting_fun(seq, cap, min(cap - len(sam_list), orig_sam_len * (max_mult - 1)),
                                                replace=True))

            # sample elements of lists to match cap
            # this is done for files with multiple sequences longer than cutlength + cap
            if len(sam_list) > cap:
                sam_list = random.sample(sam_list, cap)
            split_seq_list.extend(sam_list)

        # create kmers
        split_seq_kmer_list = list2kmer(split_seq_list, kmer)
        # create df
        cl_df_list.append(pd.DataFrame({'sequence': split_seq_kmer_list,
                                        'label': label_iter}))
        lo_counter_list.append(lo_counter)
        label_iter += 1
    # all classes together
    full_df = pd.concat(cl_df_list)
    # shuffle
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    return full_df, lo_counter_list


def ft_data_process(dirlist, name, path, cap, cutlength, kmer, filetype='train', add_info='', labels=None, max_mult=1):
    # TODO missing assert str and len of dirList
    possible_names = ["train", "dev", "validation", "test"]
    if filetype not in possible_names:
        raise ValueError("filetype must be one of the following %s" % ", ".join(possible_names))
    if filetype == "validation":
        filetype = "dev"
    # create dir
    if os.path.exists(path + "/" + name):
        location = path + "/" + name
        if os.path.exists(location + "/" + filetype + ".tsv"):
            raise ValueError("Dir %s already exists and %s data file within it. Delete beforehand"
                             % (location, filetype + ".tsv"))
        print("Warning: %s already exists, will insert data file into it, append the info file." % location)
    else:
        location = create_dir(name, path)
    lines = ["Data from dirs: " + ', '.join(dirlist),
             "Cut length: " + str(cutlength),
             "Kmer: " + str(kmer),
             "cap: %s" % cap
            ]
    if labels is not None:
        lines.append("labels %s are %s" % (labels if labels is not None else '',
                                           list(range(len(labels)))))
    lines.append(add_info + "\n")

    # write train/test file
    ft_pd, lo_counter = ft_df_creation(class_dirs=dirlist, cap=cap, cutlength=cutlength,
                                       kmer=kmer, max_mult=max_mult)
    ft_pd.to_csv(location + "/" + filetype + ".tsv", sep='\t', index=False)
    lines.extend(["%s file:" % filetype,
                  "Left out sequences due to length: %s of classes %s"
                  % (", ".join([str(x) for x in lo_counter]),
                     ", ".join(
                         [str(y) for y in (
                             labels if labels is not None else numpy.arange(len(dirlist)))])),
                  "Number of %s sub-sequences: %s" % (filetype, str(len(ft_pd.index))),
                  "By class count: \n%s" % (str(ft_pd['label'].value_counts())),
                  "\n"])
    # write info file
    create_data_info_file(location, lines)
