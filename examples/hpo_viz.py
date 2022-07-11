import matplotlib.pyplot as plt
import optuna
import joblib
import plotly
from sql_db import create_connection

import matplotlib
import matplotlib.pyplot as plt
import argparse
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--study_name",
        default=None,
        type=str,
        required=True,
        help="name of the study to fetch",
    )

    args = parser.parse_args()

    study = optuna.create_study(
        study_name="hpo_stride3_150",
        direction="maximize",
        storage=create_connection("examples/example.db"),
        load_if_exists=True
    )

    # joblib.dump(study, location + ".pkl")
    # study = joblib.load("hpo_studies/dnabert6.pkl")


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
        print("%s, Params: %s" % (t.value, t.params))



stride3 = get_study("hpo_stride3_150")
best_trial_params(stride3, 5)

dnabert6 = get_study("hpo_dnabert6_150")
best_trial_params(dnabert6, 5)

mask2 = get_study("hpo_mask8_150")
best_trial_params(mask2, 5)





    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
    plt.savefig("test.png")
    print(study.best_trial.params)
