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


