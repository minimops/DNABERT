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
                name='pt_015_eval_v2', path='created_data', kmer=6, ratio=2, low_b=12,
                add_info='sampling 10 percent of train data with ratio 2higher low bound of 12',
                perc=.015)
###########

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
                cutlength=150, max_mult=4, perc=750000, add_info='10 percent split with 1024 cap and 4 max mult')
####



