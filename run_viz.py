from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

excl1 = ["ft_hpt_v3_base_LR_trial", "ft_hpt_v3_15", "ft_hpt_v3_11", "ft_hpt_v3_16",
         "ft_hpt_v3_17", "ft_hpt_v3_17_2", "ft_hpt_v3_18", "ft_hpt_v3_19", "ft_hpt_v3_20",
         "ft_hpt_v3_21", "ft_hpt_v3_14", "ft_hpt_v3_10", "ft_hpt_v3_10_2", "ft_hpt_v3_6",
         "ft_hpt_v3_7"]
full_df3 = full_df[~full_df.run_id.isin(excl1)]

incl1 = ["ft_hpt_v3_6", "ft_hpt_v3_7", "ft_hpt_v3_14",
         "ft_hpt_v3_15", "ft_hpt_v3_19", "ft_hpt_v3_20", "ft_hpt_v3_21",
         "ft_hpt_v3_base", "ft_hpt_v3_2", "ft_hpt_v3_13"]
full_df4 = full_df[full_df.run_id.isin(incl1)]

incl2 = ["ft_hpt_v3_11", "ft_hpt_v3_16", "ft_hpt_v3_17",
         "ft_hpt_v3_17_2", "ft_hpt_v3_18", "ft_hpt_v3_10", "ft_hpt_v3_10_2",
         "ft_hpt_v3_base", "ft_hpt_v3_2", "ft_hpt_v3_13"]
full_df5 = full_df[full_df.run_id.isin(incl2)]

sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df5,
    kind="line",
    x="global_step",
    y="acc",
    hue="run_id",
    style="run_id"
)
#plt.show()
plt.savefig("plots/ft_pt_hpt_runs_stride.png")


excl = ["ft_hpt_v3_base_LR_trial", "ft_hpt_v3_1", "ft_hpt_v3_3", "ft_hpt_v3_9", "ft_hpt_v3_10", "ft_hpt_v3_8",
        "ft_hpt_v3_1"]
full_df2 = full_df[~full_df.run_id.isin(excl)]


sns.set_theme()
sns.set_style("whitegrid")
x = sns.relplot(
    data=full_df2,
    kind="line",
    x="global_step",
    y="acc",
    hue="run_id",
    style = "run_id"
)
plt.show()




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
plt.savefig("plots/pt_hpt_runs.png")

