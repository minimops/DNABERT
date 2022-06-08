import os

import pandas as pd
from data_preprocessing import seq2kmer

# read in tsv file
df = pd.read_table("created_data/ft_hpt_v3_setup_v3/interim_dev.tsv")
df["sequence"] = df.apply(lambda row: seq2kmer(row[0], 6, 3), axis=1)

df1 = pd.read_table("created_data/ft_hpt_v3_setup_v3/interim_train.tsv")
df1["sequence"] = df1.apply(lambda row: seq2kmer(row[0], 6, 3), axis=1)



os.mkdir("created_data/ft_hpt_v3_setup_v4")
df.to_csv("created_data/ft_hpt_v3_setup_v4/dev.tsv", sep="\t", index=False)
df1.to_csv("created_data/ft_hpt_v3_setup_v4/train.tsv", sep="\t", index=False)


# read in tsv file
df = pd.read_table("created_data/ft_hpt_v3_setup_v3/interim_dev.tsv")
df["sequence"] = df.apply(lambda row: seq2kmer(row[0], 6, 6), axis=1)

df1 = pd.read_table("created_data/ft_hpt_v3_setup_v3/interim_train.tsv")
df1["sequence"] = df1.apply(lambda row: seq2kmer(row[0], 6, 6), axis=1)



os.mkdir("created_data/ft_hpt_v3_setup_v5")
df.to_csv("created_data/ft_hpt_v3_setup_v5/dev.tsv", sep="\t", index=False)
df1.to_csv("created_data/ft_hpt_v3_setup_v5/train.tsv", sep="\t", index=False)