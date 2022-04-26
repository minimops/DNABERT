import numpy as np
import matplotlib.pyplot as plt
import torch
import re

# TODO this plotting stuff seems to not work at all on remote


def create_loss_dict(model_path, metric, finetune=False):

    if finetune:
        # TODO
        # placeholder
        return None
    else:
        if metric not in ['perplexity', 'eval_loss']:
            raise ValueError('Incompatible metric')
        tr_args = vars(torch.load(model_path + '/training_args.bin'))
        steps = range(0, tr_args.get('max_steps') + 1, tr_args.get('logging_steps'))
        with open(model_path + '/eval_results.txt', 'r') as f:
            res = f.readlines()
        # only keep lines with desired metric
        res_nums = []
        for r in res:
            if r. startswith(metric):
                # remove '\n'
                # remove beginning word
                res_nums.append(float(re.sub('^' + metric + ' = ', '', r.strip())))
        return dict(zip(steps, res_nums))


def plot_pt_loss(path, losses:dict, color='b'):
    plt.plot(losses.keys(), losses.values(), color, label='Training loss')
    plt.title('Pretraining Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plt.savefig(path + '/pt_loss_plot.png')


def plot_ft_loss(path, tr_losses:dict, val_losses:dict, colors=None):
    if colors is None:
        colors = ['b', 'g']
    plt.plot(tr_losses.keys(), tr_losses.values(), colors[0], label='Training loss')
    plt.plot(val_losses.keys(), val_losses.values(), colors[1], label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plt.savefig(path + '/pt_loss_plot.png')


# Usage example
# plot_pt_loss('DNABERT/examples/output_viral_test6', create_loss_dict('DNABERT/examples/output_viral_test6'))

# create_loss_dict('DNABERT/examples/output_viral_test2_6', 'eval_loss')
