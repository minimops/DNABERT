# script to be run for finetuning

from DNABERT.run_funs import *
import argparse
import os


def finetune(dirname, data_path, model_path, add_info, dirpath='DNABERT/Test_runs/pt'):
    # create dir
    location = create_dir(dirname, dirpath)
    # create info file
    create_info_file(data_path, location, add_info, phase='finetuning', model_dir=model_path)

    # run finetuning
    try:
        msg = ''
        # TODO is this really a good way to do this?
        os.system() # call the other python script here
    except Exception as inst:
        msg = str(str(type(inst)) + '\n' +
                  ', '.join(inst.args) + '\n')
    finally:
        # complete info file
        complete_info_file(location, msg)
        if 'inst' in locals():
            raise inst




        # def main():
#     parser = argparse.ArgumentParser()
#
#     # Required parameters
#     parser.add_argument(
#         "--train_data_path",
#         default=None,
#         type=str,
#         required=True,
#         help="The input training data directory path (should contain a text file named 'data_info.txt' "
#              "and the training data text file)."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         required=True,
#         help="The output directory where the model predictions and checkpoints will be written.",
#     )
#     parser.add_argument(
#         "--add_info",
#         type=str,
#         required=True,
#         help="Additional info of the model that iss being trained",
#     )
#
#     # Other parameters
#     parser.add_argument(
#         "--eval_data_file",
#         default=None,
#         type=str,
#         help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
#     )
#     args = parser.parse_args()
#     pretrain(args, )
#
#
# if __name__ == "__main__":
#     main()
#
