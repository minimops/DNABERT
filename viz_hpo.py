from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

loc = "Test_runs/hpo_test4"

dirs = []
for m in listdir(loc):
    if not m == "study.pkl":
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