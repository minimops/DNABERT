from datetime import datetime
import os
import torch
import sys
import fileinput


# creating funs to run pretraining and capturing important information


# creates a text file of information to be called when running training
def create_run_info_file(data_path, path, add_info='', phase='pretraining', model_dir=None):
    if phase not in ['pretraining', 'finetuning']:
        raise ValueError('phase should be a subset of pretraining or finetuning')
    if phase == 'finetuning' and model_dir is None:
        raise ValueError('Please provide Model directory for finetuning')

    # gpu info
    gpu_name = torch.cuda.get_device_name()
    gpu_props = str(torch.cuda.get_device_properties(torch.cuda.device))

    # get data info text file text
    # might be one up
    if data_path.endswith(".txt"):
        data_path = data_path.rsplit("/", 1)[0]
    # this requires the data to be in its separate dir and have an additional info text file
    try:
        with open(data_path + '/data_info.txt', 'r') as f:
            dat_info = f.readlines()
    except FileNotFoundError as e:
        dat_info = "No file like %s found. No data info provided" % e.filename
    # get data last changed
    lm_data = datetime.fromtimestamp(os.stat(data_path).st_mtime).strftime('%Y-%m-%d-%H:%M')

    # get current time
    time_of_start = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")

    # create text file and write
    with open(path + '/info_run.txt', 'w+') as f:
        f.write('Info file \n')
        f.write('VIR-DNABERT ' + phase + '\n\n')
        f.write('Run start time: ' + time_of_start + '\n')
        f.write('Run end time: ' + '\n\n')
        f.write('GPU used: ' + gpu_name + '\n')
        f.write(gpu_props + '\n\n')
        f.write('Data used: ' + data_path + '\n')
        f.write('last modified: ' + lm_data + '\n')
        f.write('data description:\n')
        f.write(''.join([str(i) for i in dat_info[1:]]))
        f.write('\n')
        f.write(add_info + '\n')

        if phase == 'finetuning':
            f.write('\npretrained Model dir: ' + model_dir + '\n')
            f.write('Model dir last modified: ' +
                    datetime.fromtimestamp(os.stat(model_dir).st_mtime).strftime('%Y-%m-%d-%H:%M') + '\n')
    print('Created info file at ' + path + '/info_run.txt')


# fills in training end time and runtime at the end of training
# also to be called when interrupted or errored out
def complete_run_info_file(path, msg=None):
    for line in fileinput.input(path + '/info_run.txt', inplace=True):
        if line.strip().startswith('Run end time'):
            line = 'Run end time: ' + datetime.now().strftime("%d-%b-%Y (%H:%M:%S)") + '\n'
        sys.stdout.write(line)
    if msg is not None:
        with open(path + '/info_run.txt', 'a') as f:
            f.write('\nMessage: ' + msg + '\n')
    print('Completed info file at ' + path + '/info_run.txt')


# usage:
# create_info_file('DNABERT/first_full_pt_dat', 'DNABERT/output_txt_file_test', 'more info about test run')
# complete_info_file('DNABERT/output_txt_file_test', 'test err msg')


# creates the directory for the current run files to be saved in
def create_dir(name, path=''):
    dirname = (path + '/' + name) if path else name
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("Directory ", dirname, " Created ")
    else:
        raise ValueError("Directory ", dirname, " already exists")
    return dirname


def create_data_info_file(path, info):
    # create text file and write
    dateT = datetime.now()
    with open(path + '/data_info.txt', 'w') as f:
        f.write('Data Info file \n')
        f.write(str('Created:' + str(dateT.strftime("%d-%b-%Y (%H:%M:%S)")) + '\n'))
        f.write('\n'.join(str(i) for i in info))
        f.write('\n')
    print('Created Data Info file at ' + path + '/data_info.txt')


