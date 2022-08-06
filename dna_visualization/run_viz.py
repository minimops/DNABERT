from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

loc = "../models/ft_pt_hpt"

dirs = []
for m in listdir(loc):
    dirs.append(m)

# for every model dir
df_runs = []
for m in dirs:
    dfs = []
    for file in listdir(loc + "/" + m):
        # print(file)
        if file.startswith("eval") or file.startswith("tr_args"):
            dfs.append(pd.read_csv(loc + "/" + m + "/" + file))
    df = dfs[0]
    # df = pd.merge(dfs[0], dfs[1], on="global_step")
    df["run_id"] = m
    df_runs.append(df)
full_df = pd.concat(df_runs).reset_index()
# find eval_res file
# read intro df
# add col with dir name

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df,
    kind="line",
    x="global_step",
    y="acc",
    hue="run_id",
    style="run_id"
)
plt.show()

excl1 = ["ft_hpt_v3_base_LR_trial", "ft_hpt_v3_15", "ft_hpt_v3_11", "ft_hpt_v3_16",
         "ft_hpt_v3_17", "ft_hpt_v3_17_2", "ft_hpt_v3_18", "ft_hpt_v3_19", "ft_hpt_v3_20",
         "ft_hpt_v3_21", "ft_hpt_v3_14", "ft_hpt_v3_10", "ft_hpt_v3_10_2", "ft_hpt_v3_6",
         "ft_hpt_v3_7", "ft_hpt_v3_23", "ft_hpt_v3_3", "ft_hpt_v3_1"]
full_df3 = full_df[~full_df.run_id.isin(excl1)]

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df3,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id",
    style="run_id"
)
plt.show()

maskk = ["ft_hpt_v3_6", "ft_hpt_v3_7", "ft_hpt_v3_14",
         "ft_hpt_v3_15", "ft_hpt_v3_19", "ft_hpt_v3_20", "ft_hpt_v3_21",
         "ft_hpt_v3_24", "ft_hpt_v3_26", "ft_hpt_v3_27", "ft_hpt_v3_30", "ft_hpt_v3_32",
         "ft_hpt_v3_base", "ft_hpt_v3_2", "ft_hpt_v3_13"]
full_df4 = full_df[full_df.run_id.isin(maskk)]

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df4,
    kind="line",
    x="global_step",
    y="acc",
    hue="run_id",
    style="run_id"
)
plt.show()

strides = ["ft_hpt_v3_11", "ft_hpt_v3_16",
         "ft_hpt_v3_17_2", "ft_hpt_v3_18", "ft_hpt_v3_10_2",
         "ft_hpt_v3_25", "ft_hpt_v3_28", "ft_hpt_v3_29", "ft_hpt_v3_31",
         "ft_hpt_v3_base", "ft_hpt_v3_2", "ft_hpt_v3_13",
        "ft_hpt_v3_33", "ft_hpt_v3_34", "ft_hpt_v3_35"]
full_df5 = full_df[full_df.run_id.isin(strides)]

strides2 = [
         "ft_hpt_v3_25", "ft_hpt_v3_28", "ft_hpt_v3_29", "ft_hpt_v3_31",
         "ft_hpt_v3_13",
        "ft_hpt_v3_33", "ft_hpt_v3_34", "ft_hpt_v3_35"]
full_df5_2 = full_df[full_df.run_id.isin(strides2)]

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df5_2,
    kind="line",
    x="global_step",
    y="acc",
    hue="run_id",
    style="run_id"
)
plt.show()
#plt.savefig("plots/ft_pt_hpt_runs_mask2.png")


print(sns.color_palette("tab10", 13).as_hex())

###base
incl = ["ft_hpt_v3_base", "ft_hpt_v3_1", "ft_hpt_v3_2", "ft_hpt_v3_3", "ft_hpt_v3_5",
        "ft_hpt_v3_12", "ft_hpt_v3_13", "ft_hpt_v3_22", "ft_hpt_v3_8", "ft_hpt_v3_9"]
full_df2 = full_df[full_df.run_id.isin(incl)]
r_i = full_df2.run_id.replace("ft_hpt_v3_", "trial_", regex=True)
full_df2["run_id"] = r_i
full_df2["recall"] = full_df["recall"] * 100
full_df2 = full_df2.sort_values('run_id')

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df2,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id",
    facet_kws={'legend_out': False}
)

x.set(ylabel='Recall in %', xlabel='Steps')
x.fig.suptitle("virBERT base", size=16)
x.fig.subplots_adjust(top=.9)
plt.ylim([60, 76])
plt.legend([],[], frameon=False)
plt.savefig("plots/trial_bases_noleg.png")
plt.show()
###


###mask
incl = ["ft_hpt_v3_7", "ft_hpt_v3_14", "ft_hpt_v3_15", "ft_hpt_v3_19", "ft_hpt_v3_20", "ft_hpt_v3_21",
        "ft_hpt_v3_26", "ft_hpt_v3_30"]
full_df2 = full_df[full_df.run_id.isin(incl)]
r_i = full_df2.run_id.replace("ft_hpt_v3_", "trial_", regex=True)
full_df2["run_id"] = r_i
full_df2["recall"] = full_df["recall"] * 100
full_df2 = full_df2.sort_values('run_id')

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df2,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id",
    facet_kws={'legend_out': False}
)

x.set(ylabel='Recall in %', xlabel='Steps')
x.fig.suptitle("virBERT-maskX", size=16)
x.fig.subplots_adjust(top=.9)
plt.ylim([60, 76])
plt.legend([],[], frameon=False)
plt.savefig("plots/trial_mask_noleg.png")
plt.show()
###


###stride
incl = ["ft_hpt_v3_11", "ft_hpt_v3_16", "ft_hpt_v3_17_2", "ft_hpt_v3_18",
        "ft_hpt_v3_25", "ft_hpt_v3_28", "ft_hpt_v3_29", "ft_hpt_v3_31", "ft_hpt_v3_33"]
full_df2 = full_df[full_df.run_id.isin(incl)]
r_i = full_df2.run_id.replace("ft_hpt_v3_", "trial_", regex=True)
r_i = r_i.replace("17_2", "17")
full_df2["run_id"] = r_i
full_df2["recall"] = full_df["recall"] * 100
full_df2 = full_df2.sort_values('run_id')

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df2,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id",
    facet_kws={'legend_out': False}
)

x.set(ylabel='Recall in %', xlabel='Steps')
x.fig.suptitle("virBERT-strideX", size=16)
x.fig.subplots_adjust(top=.9)
plt.ylim([60, 76])
plt.legend([],[], frameon=False)
plt.savefig("plots/trial_stride_noleg.png")
plt.show()
###

show = ["ft_hpt_v3_base", "ft_hpt_v3_7", "ft_hpt_v3_30",
       "ft_hpt_v3_13",
        "ft_hpt_v3_2", "ft_hpt_v3_6"]
full_dfX = full_df[full_df.run_id.isin(show)]


sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_dfX,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id"
)
plt.show()


show = ["ft_hpt_v3_base", "ft_hpt_v3_13", "ft_hpt_v3_30",
       "ft_hpt_v3_35"] # , "ft_hpt_v3_33", "ft_hpt_v3_31"]
full_dfX = full_df[full_df.run_id.isin(show)]
r_i = full_dfX.run_id.replace("ft_hpt_v3_", "", regex=True)
r_i = r_i.replace("13", "base_highLR", regex=True)
r_i = r_i.replace("30", "mask8", regex=True)
r_i = r_i.replace("35", "stride3", regex=True)
full_dfX["run_id"] = r_i
full_dfX["recall"] = full_df["recall"] * 100

palette ={"mask8": "C0", "stride3": "C1", "base": "C2", "base_highLR": "C3"}

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_dfX,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id",
    palette=palette,
    facet_kws={'legend_out': False}
)
x.set(ylabel='Recall in %', xlabel='Steps')
x.fig.suptitle("Trial Finetuning", size =16)
x.fig.subplots_adjust(top=.9)
leg = x.axes.flat[0].get_legend()
leg.set(bbox_to_anchor=(.495, .695))
new_title = 'trial'
leg.set_title(new_title)
# fig, ax = plt.subplots()
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda a, pos: '{:,.2f}'.format(a/1000) + 'K'))
# plt.show()
plt.savefig("plots/ft_trials.png")
####


loc = "../models/pt_hpt"

dirs = []
for m in listdir(loc):
    dirs.append(m)

# for every model dir
df_runs = []
for m in dirs:
    dfs = []
    for file in listdir(loc + "/" + m):
        # print(file)
        if file.startswith("eval") or file.startswith("tr_args"):
            dfs.append(pd.read_csv(loc + "/" + m + "/" + file))
    df = dfs[0]
    # df = pd.merge(dfs[0], dfs[1], on="global_step")
    df["run_id"] = m
    df_runs.append(df)
full_df = pd.concat(df_runs).reset_index()
# find eval_res file
# read intro df
# add col with dir name


sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df,
    kind="line",
    x="global_step",
    y="eval_loss",
    hue="run_id",
    style="run_id"
)
plt.show()
plt.savefig("plots/pt_hpt_runs.png")


show = ["pt_hpt_v3_base_t2", "pt_hpt_v3_13", "pt_hpt_v3_30",
       "pt_hpt_v3_35"] # , "ft_hpt_v3_33", "ft_hpt_v3_31"]
full_dfX2 = full_df[full_df.run_id.isin(show)]
r_i = full_dfX2.run_id.replace("pt_hpt_v3_", "", regex=True)
r_i = r_i.replace("13", "base_highLR", regex=True)
r_i = r_i.replace("_t2", "", regex=True)
r_i = r_i.replace("30", "mask8", regex=True)
r_i = r_i.replace("35", "stride3", regex=True)
full_dfX2["run_id"] = r_i

palette ={"mask8": "C0", "stride3": "C1", "base": "C2", "base_highLR": "C3"}

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_dfX2,
    kind="line",
    x="global_step",
    y="eval_loss",
    hue="run_id",
    palette=palette,
    facet_kws={'legend_out': False}
)
x.set(ylabel='Validation Loss', xlabel='Steps')
x.fig.suptitle("Trial Pretraining", size =16)
x.fig.subplots_adjust(top=.9)
leg = x.axes.flat[0].get_legend()
leg.set(bbox_to_anchor=(.495, .595))
new_title = 'trial'
leg.set_title(new_title)
# plt.show()
plt.savefig("plots/pt_trial.png")
####





########
loc = "../models/pt_virbert"

# for every model dir
df_runs = []
for file in listdir(loc):
    # print(file)
    df = pd.read_csv(loc + "/" + file)
    df["run_id"] = file[:-4]
    df_runs.append(df)
full_df = pd.concat(df_runs).reset_index(drop=True)

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df[full_df.global_step <= 100000],
    kind="line",
    x="global_step",
    y="eval_loss",
    hue="run_id",
    facet_kws={'legend_out': False}
)
x.set(ylabel='Validation Loss', xlabel='Steps')
x.fig.suptitle("Virbert Pretraining Loss", size =16)
x.fig.subplots_adjust(top=.9)
leg = x.axes.flat[0].get_legend()
leg.set(bbox_to_anchor=(.925, .895))
new_title = 'model'
leg.set_title(new_title)
plt.show()
plt.savefig("plots/pt_val_loss.png")
####


########
loc = "../models/fin_ft/"

dirs = []
for m in listdir(loc):
    dirs.append(m)

# for every model dir
df_runs = []
for m in dirs:
    dfs = []
    for file in listdir(loc + "/" + m):
        # print(file)
        if file.startswith("eval") or file.startswith("tr_args"):
            dfs.append(pd.read_csv(loc + "/" + m + "/" + file))
    df = dfs[0]
    # df = pd.merge(dfs[0], dfs[1], on="global_step")
    df["run_id"] = m
    df_runs.append(df)
full_df = pd.concat(df_runs).reset_index(drop=True)
# find eval_res file
# read intro df
# add col with dir name


le_df = full_df[full_df.run_id.str.contains("le_1k")][full_df.global_step <= 100000].reset_index(drop=True)
le_df.run_id = [s[:-6] for s in le_df.run_id]
le_df["recall"] = le_df["recall"] * 100
sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=le_df,
    kind="line",
    x="global_step",
    y="recall",
    hue="run_id",
    facet_kws={'legend_out': False}
)
x.set(ylabel='Recall in %', xlabel='Steps')
x.fig.suptitle("Training under Linear Evaluation", size =16)
x.fig.subplots_adjust(top=.9)
leg = x.axes.flat[0].get_legend()
leg.set(bbox_to_anchor=(.94, .43))
new_title = 'model'
leg.set_title(new_title)
# plt.show()
plt.savefig("plots/ft_le_recall.png")
####
