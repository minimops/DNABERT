import matplotlib.pyplot as plt
import optuna
import joblib
import plotly
from sql_db import create_connection

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


def get_study(study_name, storage_loc="examples/example.db"):

    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=create_connection(storage_loc),
        load_if_exists=True
    )


def best_trial_params(study, n):
    t_list = study.trials
    for t in t_list:
        if t.value is None:
            if print(type(t.values)) is not None and len(t.values) > 0 and not all(v is None for v in t.values):
                t.value = max(t.values)
            else:
                t.value = 0.5
    t_list.sort(key=lambda x: x.value)
    for t in t_list[-n:]:
        print("trial%s: %s with %s rep steps, Params: %s" % (t.number, t.value, len(t.intermediate_values), t.params))


def print_trial_vals(study, number):
    print(study.trials[number].intermediate_values)


stride3 = get_study("hpo_stride3_150_v3")
best_trial_params(stride3, 12)
print_trial_vals(stride3, 3)

optuna.visualization.matplotlib.plot_optimization_history(stride3)
plt.show()
optuna.visualization.matplotlib.plot_param_importances(stride3)
plt.show()

virbert = get_study("hpo_virbert_150_v2")
best_trial_params(virbert, 15)

optuna.visualization.matplotlib.plot_optimization_history(virbert)
plt.show()

sns.set_theme()
sns.set_style("whitegrid")
x = optuna.visualization.matplotlib.plot_param_importances(virbert)

plt.tight_layout()
plt.savefig('plots/hpo_importance.png')
plt.show()


mask2 = get_study("hpo_mask8_150_final")
best_trial_params(mask2, 12)

optuna.visualization.matplotlib.plot_optimization_history(mask2)
plt.show()
optuna.visualization.matplotlib.plot_param_importances(mask2)
plt.show()


dnabert6 = get_study("hpo_dnabert6_150")
best_trial_params(dnabert6, 8)

optuna.visualization.matplotlib.plot_optimization_history(dnabert6)
plt.show()
optuna.visualization.matplotlib.plot_param_importances(dnabert6)
plt.show()



stride3_1k = get_study("hpo_stride3_1000")
best_trial_params(stride3_1k, 15)

optuna.visualization.matplotlib.plot_optimization_history(stride3_1k)
plt.show()
optuna.visualization.matplotlib.plot_param_importances(stride3_1k)
plt.show()

