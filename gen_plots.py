import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# function that reads eval_results and makes a pandas.df out of it
def create_metric_df(paths):
    dfs = []
    for p in paths:
        try:
            eval_path = p + "/eval_results.csv"
            train_path = p + "/tr_args.csv"
            runid = p.rsplit("/", 1)[1]
            df1 = pd.read_csv(eval_path, header=0)
            df2 = pd.read_csv(train_path, header=0)
            df = pd.concat([df1, df2], ignore_index=True, sort=False)
            df["type"] = runid
            dfs.append(df)
        except FileNotFoundError as e:
            print(e)
            print("No eval or training files in dir %s" % p)
            print("Skipping this run.")
            continue

    return pd.concat(dfs).reset_index()


# function that creates a seaborn line plot metriv vs global_step
# and saves it to path as png
def plot_measure(data: pd.DataFrame, pl_metric, path=None, zero_xlim=True):
    if pl_metric not in data.columns:
        raise ValueError("Desired Metric %s not supplied in data." % pl_metric)

    sns.set_theme()
    sns.set_style("whitegrid")
    x = sns.relplot(
        data=data,
        kind="line",
        x="global_step",
        y=pl_metric,
        hue="type"
    )
    if zero_xlim:
            x.set(ylim=0)
    plt.show()
    if path is not None:
        plt.savefig(path)


def plot_losses(data: pd.DataFrame, cols, path=None, zero_xlim=True):
    cols.append("global_step")
    df = data[cols]
    df = df.melt('global_step', var_name='Type', value_name='Loss', ignore_index=True).dropna().reset_index()

    sns.set_theme()
    sns.set_style("whitegrid")
    x = sns.relplot(
        data=df,
        kind="line",
        x="global_step",
        y="Loss",
        hue="Type"
    )
    if zero_xlim:
            x.set(ylim=0)
    plt.show()
    if path is not None:
        plt.savefig(path)


# Usage
pt_runs = ["Test_runs/" + x for x in os.listdir("Test_runs") if x.startswith("pt")]
training_metrics_pt = create_metric_df(pt_runs)
plot_measure(training_metrics_pt, "eval_loss", "plots/pt_first_tests_1_percent_eval_loss.png")
plot_measure(training_metrics_pt, "perplexity", "plots/pt_first_tests_1_percent_perplexity.png")

ft_runs = ["Test_runs/" + x for x in os.listdir("Test_runs") if x.startswith("ft")]
training_metrics_ft = create_metric_df(ft_runs)

for metric in training_metrics_ft.columns[2:]:
    plot_measure(training_metrics_ft, metric, "plots/ft_first_tests_1_percent_%s.png" % metric, False)

# testing plotting both training and eval loss
plot_losses(create_metric_df(["Test_runs/ft_1_percent_new_changes"]), ["training_loss", "eval_loss"])

data = create_metric_df(["Test_runs/ft_1_percent_new_changes"])
cols = ["training_loss", "eval_loss"]