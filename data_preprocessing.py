import random
import numpy.random
import pandas as pd
from Bio import SeqIO
from statistics import mean, median
import os
from math import ceil
# from motif.motif_utils import seq2kmer
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
            if diff < low_b:
                break
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
def split_sequences_rand(seq_list, low_b=5, upp_b=510, rat_max=.5, ratio=1):
    split_seq_list = []

    # expected average seq length
    expL = ((((upp_b - low_b) / 2) * (1 - rat_max)) + upp_b * rat_max)
    # for each sequence
    for seq in seq_list:
        if len(seq) < low_b:
            print("sequence skipped because of length < %s" % low_b)
            continue
        # ceiling here to create at least some for short seqs
        amt = int(ratio * ceil(len(seq) / expL))
        for i in range(amt):
            cutLength = sample_cut_length(low_b, min(upp_b, len(seq)), rat_max)
            if cutLength < low_b:
                continue
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
def list2kmer(spl_sequences, k, s):
    fin = []
    for spl in spl_sequences:
        fin.append(seq2kmer(spl, k, s))
    return fin


# remove genomes with 'N' entries
def removeNAseq(seqList):
    for seq in seqList:
        if "N" in seq:
            seqList.remove(seq)
    return seqList


def seq2kmer(seq, k, stride=1):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    if stride > 1:
        seq = seq[:-stride]
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)[::stride]]
    kmers = " ".join(kmer)
    return kmers


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


def seq_sub_sample(d1, d2, perc):
    seqs = []
    for d in [d1, d2]:
        seqs.append(numpy.array(d)[numpy.random.choice(len(d), int(perc * len(d)), replace=False)])
    return list(seqs[0]), list(seqs[1])


# count upper bound to give to pt_data_process for a given number of kmers desired,
# respective of k and stride
def calc_upp_bound(n_mer, k, s):
    return (n_mer - 1) * s + k


def pt_data_process(dirs_list, name, path, kmer, add_info='', low_b=5, upp_b=510, rat_max=.5, ratio=1,
                    perc=None, perc2=None, s=1, do_no=True, do_samp=True):
    location = create_dir(name, path)
    lines = []
    split_seqs_both = []
    # TODO think this is really inefficient cuz of a lot of moving memory, fix could be simple
    for d in dirs_list:
        seqs = removeNAseq(parse_fasta(d))
        n_o_cut = split_sequences_no(seqs, low_b, upp_b, rat_max)
        samp_cut = split_sequences_rand(seqs, low_b, upp_b, rat_max, ratio)
        if perc is not None:
            n_o_cut, samp_cut = seq_sub_sample(n_o_cut, samp_cut, perc)

        # TODO this is obvsl bad, no need to do this after not before
        # fix would be do no and samp seperately with condition
        if do_no:
            split_seqs_both.extend(n_o_cut)
        else:
            n_o_cut = []
        if do_samp:
            split_seqs_both.extend(samp_cut)
        else:
            samp_cut = []

        lines.extend(["Data from dir: " + d,
                      "Number of non NA sequences: " + str(len(seqs)),
                      "Median length of seq: " + str(median(map(len, seqs))),
                      "Average length of seq: " + str(mean(map(len, seqs))),
                      "Split into sequences of length: " + str(low_b) + " - " + str(upp_b),
                      "With bias probability to max len of: " + str(rat_max),
                      "Sample ratio of: " + str(ratio),
                      "Number of non-overlap sub-seqs: " + str(len(n_o_cut)),
                      "Number of sampled sub-seqs: " + str(len(samp_cut)),
                      "Kmer: " + str(kmer)
                      ])

    lines.append("\n" + add_info)

    random.shuffle(split_seqs_both)
    if perc2 is not None:
        split_seqs_both = random.sample(split_seqs_both, perc2)

    # write as interim text file
    textfile = open(location + '/split_interim.txt', "w")
    for element in split_seqs_both:
        textfile.write(element + "\n")
    textfile.close()

    # create kmers
    kmer_seqs_all = list2kmer(split_seqs_both, kmer, s)

    # write final kmers as text file
    textfile = open(location + '/full_kmers.txt', "w")
    for element in kmer_seqs_all:
        textfile.write(element + "\n")
    textfile.close()

    # write info file
    create_data_info_file(location, lines)


def ft_df_creation(class_dirs, cap, cutlength, kmer, max_mult=1, perc=None, perc2=None):
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

        if perc is not None:
            if perc > 1:
                print("'perc' is >1, interpreting it as number of examples instead of percentage.")
                if perc > len(split_seq_list) or not isinstance(perc, int):
                    raise ValueError(
                        "'perc' is greater than total examples or not an integer. Use perc to subsample not oversample here")

            split_seq_list = list(numpy.array(split_seq_list)[numpy.random.choice(len(split_seq_list),
                                                                                  (int(perc * len(
                                                                                      split_seq_list)) if perc < 1 else perc),
                                                                                  replace=False)])

        # create kmers
        # split_seq_kmer_list = list2kmer(split_seq_list, kmer)
        # create df
        df = pd.DataFrame({'sequence': split_seq_list,
                                        'label': label_iter})
        if perc2 is not None:
            df = df.sample(perc2).reset_index()
        cl_df_list.append(df)
        lo_counter_list.append(lo_counter)
        label_iter += 1
    # all classes together
    full_df = pd.concat(cl_df_list)
    # shuffle
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    # if perc2 is not None:
    #     full_df = full_df.sample(perc2).reset_index()
    return full_df, lo_counter_list


def ft_data_process(dirlist, name, path, cap, cutlength, kmer, filetype='train', add_info='', labels=None, max_mult=1,
                    perc=None, perc2=None, s=1):
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
             "cap: %s" % cap,
             "max_mult: %s" % max_mult
             ]
    if labels is not None:
        lines.append("labels %s are %s" % (labels if labels is not None else '',
                                           list(range(len(labels)))))
    lines.append(add_info + "\n")

    # write train/test file
    ft_pd, lo_counter = ft_df_creation(class_dirs=dirlist, cap=cap, cutlength=cutlength,
                                       kmer=kmer, max_mult=max_mult, perc=perc, perc2=perc2)
    ft_pd.to_csv(location + "/" + "interim_" + filetype + ".tsv", sep='\t', index=False)

    ft_pd["sequence"] = ft_pd.apply(lambda row: seq2kmer(row[0], kmer, s), axis=1)
    ft_pd.to_csv(location + "/" +  filetype + ".tsv", sep='\t', index=False)

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
