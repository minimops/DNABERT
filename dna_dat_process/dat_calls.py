from data_preprocessing import ft_data_process, pt_data_process, token_dat, calc_upp_bound, pred_data_process, seq2kmer
import pandas as pd

### calls of preprocessing funs to create data ###


## finetune with the 10 percent train split and 10 percent of the valiation data. ##
## cap at 2048 and max_mult at 8. Creates a 5.4G train file and a 1.6G val file. ##

ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_10percent", "../data/viral-phage_1_3/train_10percent"],
                name='ft_10_2048', path='created_data', cap=2048, kmer=6, filetype="train",
                cutlength=150, max_mult=8, add_info='10 percent split with 1024 cap and 4 max mult')

ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_10_2048', path='created_data', cap=2048, kmer=6, filetype="dev",
                cutlength=150, max_mult=8, perc=.10, add_info='10 percent of val data 2048 cap, 8 max mult')

###########

#########
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v2', path='created_data', kmer=6, ratio=2, low_b=12, perc=0.1,
                add_info='10 percent of train dataset with higher lower bound of 12')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v9', path='created_data', kmer=6, ratio=2, low_b=36, perc=0.1,
                add_info='10 percent of train dataset with higher lower bound of 12')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v11', path='created_data', kmer=6, ratio=3, low_b=36, perc=0.1,
                do_no=False, perc2=776000,
                add_info='10 percent of train dataset with higher lower bound of 36, only sampling')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_015_eval_v2', path='created_data', kmer=6, ratio=2, low_b=12,
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 12',
                perc=.015)
###########
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v13', path='created_data', kmer=6, ratio=2, low_b=36, perc=0.1, rat_max=.3,
                add_info='10 percent of train dataset with higher lower bound of 12')
#########
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v3', path='created_data', kmer=6, ratio=3, low_b=12, perc=0.1,
                add_info='10 percent of train dataset with higher lower bound of 12, ratio of 3')
########

#########
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v4', path='created_data', kmer=6, ratio=2, low_b=12, perc=0.1, s=3, upp_b=calc_upp_bound(505, 6, 3),
                add_info='10 percent of train dataset with higher lower bound of 12')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v5', path='created_data', kmer=6, ratio=2, low_b=12, perc=0.28, s=3, upp_b=calc_upp_bound(505, 6, 3),
                add_info='25 percent of train dataset with higher lower bound of 12')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v7', path='created_data', kmer=6, ratio=8, low_b=12, perc=0.1, s=3, upp_b=calc_upp_bound(505, 6, 3),
                perc2=772000, add_info='25 percent of train dataset with higher lower bound of 12')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_015_eval_v4', path='created_data', kmer=6, ratio=2, low_b=12, s=3, upp_b=calc_upp_bound(505, 6, 3),
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 12',
                perc=.015)
###########
########

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v8', path='created_data', kmer=6, ratio=17, low_b=12, perc=0.1, s=6, upp_b=calc_upp_bound(505, 6, 6),
                perc2=772000, add_info='10 percent of train dataset with higher lower bound of 12')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_015_eval_v5', path='created_data', kmer=6, ratio=4, low_b=12, s=6, upp_b=calc_upp_bound(505, 6, 6),
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 12',
                perc=.015)
###########


pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v10', path='created_data', kmer=6, ratio=5, low_b=36, perc=0.1, s=3, upp_b=1000,
                perc2=772000, add_info='lower bound of 12, upper of 1000')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_015_eval_v10', path='created_data', kmer=6, ratio=4, low_b=12, s=3, upp_b=1000,
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 12',
                perc=.015)
###########

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v12', path='created_data', kmer=6, ratio=2, low_b=36, perc=0.1, s=3, upp_b=510,
                perc2=772000, add_info='lower bound of 36, upper of 510')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_015_eval_v12', path='created_data', kmer=6, ratio=2, low_b=36, s=3, upp_b=510,
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 36',
                perc=.015)
###########
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_train_v14', path='created_data', kmer=6, ratio=5, low_b=36, perc=0.1, s=3, upp_b=1000, rat_max=.3,
                perc2=772000, add_info='lower bound of 36, upper of 1000, bias of .3')
########


#####
ft_data_process(dirlist=["../data/viral-no-phage_1_3/train", "../data/viral-phage_1_3/train"],
                name='ft_hpt_v3_setup_v2', path='created_data', cap=1024, kmer=6, filetype="train",
                cutlength=150, max_mult=4, perc=2000000 , add_info='10 percent split with 1024 cap and 4 max mult')


ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_hpt_v3_setup_v2', path='created_data', cap=1024, kmer=6, filetype="dev",
                cutlength=150, max_mult=4, perc=1000000, add_info='10 percent split with 1024 cap and 4 max mult')
####


#####
ft_data_process(dirlist=["../data/viral-no-phage_1_3/train", "../data/viral-phage_1_3/train"],
                name='ft_hpt_v3_setup_v3', path='created_data', cap=1024, kmer=6, filetype="train",
                cutlength=150, max_mult=4, perc=2000000, add_info='10 percent split with 1024 cap and 4 max mult')


ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_hpt_v3_setup_v3', path='created_data', cap=1024, kmer=6, filetype="dev",
                cutlength=150, max_mult=4, perc=750000, add_info    ='10 percent split with 1024 cap and 4 max mult')
####
#####
ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_10percent", "../data/viral-phage_1_3/train_10percent"],
                name='ft_hpo_v1', path='created_data', cap=2048, kmer=6, filetype="train",
                cutlength=150, max_mult=2, perc=1500000,
                add_info='')


ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_hpo_dev', path='created_data', cap=256, kmer=6, filetype="dev",
                cutlength=150, max_mult=1, perc=500000, add_info='')
####



####
ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_hpo', path='created_data', cap=2048, kmer=6, filetype="dev",
                cutlength=150, max_mult=5, perc=500000, add_info='1M examples with 1024 cap and 4 max mult')
####


ft_data_process(dirlist=["../data/viral-no-phage_1_3/test", "../data/viral-phage_1_3/test"],
                name='pred_test_setup', path='created_data', cap=1024, kmer=6, filetype="dev",
                cutlength=150, max_mult=4, perc=1000000, add_info='10 percent split with 1024 cap and 4 max mult')



#######
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_full_train_v9', path='created_data', kmer=6, ratio=2, low_b=36,
                add_info='full dataset for pretraining, created like v9 test dataset')
#######
pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_eval_v2', path='created_data', kmer=6, ratio=2, low_b=36,
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 12',
                perc=.1)
#######

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_stride_full', path='created_data', kmer=6, ratio=4.7, low_b=36, s=3, upp_b=1000,
                add_info='full stride 3 training data, lower bound of 36, upper of 1000')

pt_data_process(dirs_list=["../data/viral-no-phage_1_3/train",
                           "../data/viral-phage_1_3/train"],
                name='pt_10_eval_stride', path='created_data', kmer=6, ratio=4.7, low_b=36, s=3, upp_b=1000, perc=.1,
                add_info='full stride 3 eval data, lower bound of 36, upper of 1000')

#####
pred_150 = pred_data_process(["../data/viral-no-phage_1_3/test", "../data/viral-phage_1_3/test"],
                            cutlength=150, cap=64)
token_dat(pred_150, name="pred_150_s1", path="created_data", kmer=6, s=1)
token_dat(pred_150, name="pred_150_s3", path="created_data", kmer=6, s=3)


pred_1k = pred_data_process(["../data/viral-no-phage_1_3/test", "../data/viral-phage_1_3/test"],
                            cutlength=1000, cap=64)
token_dat(pred_1k, name="pred_1k_s1", path="created_data", kmer=6, s=1)
token_dat(pred_1k, name="pred_1k_s3", path="created_data", kmer=6, s=3)
###

################################
# 150nt
#####
# ~8M examples for 10percent of labels finetuning
dfs_10per = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_10percent", "../data/viral-phage_1_3/train_10percent"],
                name='ft_10p_s1', path='created_data', cap=4500, kmer=6, filetype="train",
                cutlength=150, max_mult=2, perc=4194304,
                add_info='')

dfs2 = dfs_10per
import pickle
with open('created_data/ft_10_150.pkl', 'wb') as outp:
    pickle.dump(dfs2, outp, pickle.HIGHEST_PROTOCOL)

# creates ~2M hpo examples
num = 1048576 # 2^x
dfs = []
for df in dfs_10per:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=1), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10/" + "train.tsv", sep='\t', index=False)


# ~1.2M examples to validate on
dfs_10_dev = ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_10p_s1', path='created_data', cap=256, kmer=6, filetype="dev",
                cutlength=150, max_mult=1, perc=614400,
                add_info='')

dfs3 = dfs_10_dev
import pickle
with open('created_data/ft_150_dev.pkl', 'wb') as outp:
    pickle.dump(dfs3, outp, pickle.HIGHEST_PROTOCOL)

# creates ~750k examples for hpo
num = 378880 # *2 divisbible by 2048
dfs = []
for df in dfs_10_dev:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=1), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10/" + "dev.tsv", sep='\t', index=False)
####




##### 10 percent stride 3
# ~8.4M examples for 10percent of labels finetuning
dfs_10per_stride3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_10percent", "../data/viral-phage_1_3/train_10percent"],
                name='ft_10p_s3', path='created_data', cap=4500, kmer=6, filetype="train",
                cutlength=150, s=3, max_mult=2, perc=4194304,
                add_info='')


dfs4 = dfs_10per_stride3
import pickle
with open('created_data/ft_10_s3_150.pkl', 'wb') as outp:
    pickle.dump(dfs4, outp, pickle.HIGHEST_PROTOCOL)

# creates ~2M hpo examples
num = 1048576 # 2^x
dfs = []
for df in dfs_10per_stride3:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=3), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10_s3/" + "train.tsv", sep='\t', index=False)


# ~1.2M examples to validate on
dfs_10_s3_dev = ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_10p_s3', path='created_data', cap=256, kmer=6, filetype="dev",
                cutlength=150, s=3, max_mult=1, perc=614400,
                add_info='')

dfs5 = dfs_10_s3_dev
import pickle
with open('created_data/ft_150_s3_dev.pkl', 'wb') as outp:
    pickle.dump(dfs5, outp, pickle.HIGHEST_PROTOCOL)

# creates ~750k examples for hpo
num = 378880 # *2 divisbible by 2048
dfs = []
for df in dfs_10_s3_dev:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=3), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10_s3/" + "dev.tsv", sep='\t', index=False)
####



##### 1 percent 150nt
# ~2M examples for 1percent of labels finetuning
dfs_1per_150nt = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_1percent", "../data/viral-phage_1_3/train_1percent"],
                name='ft_1p_s1', path='created_data', cap=10800, kmer=6, filetype="train",
                cutlength=150, max_mult=3, perc=1048576,
                add_info='')

# stride 3
# ~2M examples for 1percent of labels finetuning
dfs_1per_150nt_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_1percent", "../data/viral-phage_1_3/train_1percent"],
                name='ft_1p_s3', path='created_data', cap=10800, kmer=6, filetype="train",
                cutlength=150, s=3, max_mult=3, perc=1048576,
                add_info='')

##### 0.1 percent 150nt
# ~512k examples for 0.1percent of labels finetuning
dfs_01per_150nt = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train0_1percent", "../data/viral-phage_1_3/train0_1percent"],
                name='ft_01p_s1', path='created_data', cap=30000, kmer=6, filetype="train",
                cutlength=150, max_mult=3, perc=262144,
                add_info='')

# stride 3
# ~512k examples for 0.1percent of labels finetuning
dfs_01per_150nt_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train0_1percent", "../data/viral-phage_1_3/train0_1percent"],
                name='ft_01p_s3', path='created_data', cap=30000, kmer=6, filetype="train",
                cutlength=150, s=3, max_mult=3, perc=262144,
                add_info='')




########### 1000nt

# ~2M examples for 10percent of labels finetuning
dfs_10per_1k = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_10percent", "../data/viral-phage_1_3/train_10percent"],
                name='ft_10_1k_s1', path='created_data', cap=1200, kmer=6, filetype="train",
                cutlength=1000, max_mult=1, perc=1048576,
                add_info='')

dfsX1 = dfs_10per_1k
# import pickle
# with open('created_data/ft_10_150.pkl', 'wb') as outp:
#     pickle.dump(dfs2, outp, pickle.HIGHEST_PROTOCOL)

# creates ~500k hpo examples
num = 262144 # 2^x
dfs = []
for df in dfsX1:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=1), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10_1k/" + "train.tsv", sep='\t', index=False)


# ~600k examples to validate on
dfs_10_dev_1k = ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='r', path='created_data', cap=128, kmer=6, filetype="dev",
                cutlength=1000, max_mult=1, perc=307200,
                add_info='')

dfsX2 = dfs_10_dev_1k
# import pickle
# with open('created_data/ft_150_dev.pkl', 'wb') as outp:
#     pickle.dump(dfs3, outp, pickle.HIGHEST_PROTOCOL)

# creates ~280k examples for hpo
num = 143360 # *2 divisbible by 2048
dfs = []
for df in dfsX2:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=1), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/rep/" + "dev.tsv", sep='\t', index=False)


#### stride 3
# ~2M examples for 10percent of labels finetuning
dfs_10per_1k_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_10percent", "../data/viral-phage_1_3/train_10percent"],
                name='ft_10_1k_s3', path='created_data', cap=1200, kmer=6, filetype="train",
                cutlength=1000, s=3, max_mult=1, perc=1048576,
                add_info='')

dfsX3 = dfs_10per_1k_s3
# import pickle
# with open('created_data/ft_10_150.pkl', 'wb') as outp:
#     pickle.dump(dfs2, outp, pickle.HIGHEST_PROTOCOL)

# creates ~500k hpo examples
num = 262144 # 2^x
dfs = []
for df in dfsX3:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=3), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10_1k_s3/" + "train.tsv", sep='\t', index=False)


# ~600k examples to validate on
dfs_10_dev_1k_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/validation", "../data/viral-phage_1_3/validation"],
                name='ft_10_1k_s3', path='created_data', cap=128, kmer=6, filetype="dev",
                cutlength=1000, s=3, max_mult=1, perc=307200,
                add_info='')

dfsX4 = dfs_10_dev_1k_s3
# import pickle
# with open('created_data/ft_150_dev.pkl', 'wb') as outp:
#     pickle.dump(dfs3, outp, pickle.HIGHEST_PROTOCOL)

# creates ~280k examples for hpo
num = 143360 # *2 divisbible by 2048
dfs = []
for df in dfsX4:
    df = df.sample(num).reset_index(drop=True)
    df["sequence"] = df.apply(lambda row: seq2kmer(row[0], k=6, stride=3), axis=1)
    dfs.append(df)
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("created_data/ft_hpo_10_1k_s3/" + "dev.tsv", sep='\t', index=False)


##### 1 percent 1k
# ~524k examples for 1percent of labels finetuning
dfs_1per_1k = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_1percent", "../data/viral-phage_1_3/train_1percent"],
                name='ft_1p_1k_s1', path='created_data', cap=2800, kmer=6, filetype="train",
                cutlength=1000, max_mult=1, perc=262144,
                add_info='')

# stride 3
dfs_1per_1k_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train_1percent", "../data/viral-phage_1_3/train_1percent"],
                name='ft_1p_1k_s3', path='created_data', cap=2800, kmer=6, filetype="train",
                cutlength=1000, s=3, max_mult=1, perc=262144,
                add_info='')

##### 0.1 percent 1k
# ~166k examples for 0.1percent of labels finetuning
dfs_01per_1k = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train0_1percent", "../data/viral-phage_1_3/train0_1percent"],
                name='ft_01p_1k_s1', path='created_data', cap=10000, kmer=6, filetype="train",
                cutlength=1000, max_mult=1, perc=83200,
                add_info='')

# stride 3
dfs_01per_1k_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train0_1percent", "../data/viral-phage_1_3/train0_1percent"],
                name='ft_01p_1k_s3', path='created_data', cap=10000, kmer=6, filetype="train",
                cutlength=1000, s=3, max_mult=1, perc=83200,
                add_info='')


### 100% lables for lin eval
dfs_full_s1 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train", "../data/viral-phage_1_3/train"],
                name='ft_full_s1', path='created_data', cap=512, kmer=6, filetype="train",
                cutlength=150, max_mult=1, perc=4194304,
                add_info='')

dfs_full_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train", "../data/viral-phage_1_3/train"],
                name='ft_full_s3', path='created_data', cap=512, kmer=6, filetype="train",
                cutlength=150, s=3, max_mult=1, perc=4194304,
                add_info='')

### 100% lables for lin eval 1k
# ~2M
dfs_full_1k_s1 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train", "../data/viral-phage_1_3/train"],
                name='ft_full_1k_s1', path='created_data', cap=128, kmer=6, filetype="train",
                cutlength=1000, max_mult=1, perc=1048576,
                add_info='')

dfs_full_1k_s3 = ft_data_process(dirlist=["../data/viral-no-phage_1_3/train", "../data/viral-phage_1_3/train"],
                name='ft_full_1k_s3', path='created_data', cap=128, kmer=6, filetype="train",
                cutlength=1000, s=3, max_mult=1, perc=1048576,
                add_info='')