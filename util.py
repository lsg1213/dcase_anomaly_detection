"""
Python script for definition of utility functions.

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import csv
import glob
import itertools
import os
import re

import torch
import numpy as np
import yaml

__VERSIONS__ = "1.0.0"


def command_line_chk():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Without option argument, it will not run properly."
    )
    parser.add_argument(
        "-v", "--version", action="store_true", help="show application version"
    )
    parser.add_argument("-d", "--dev", action="store_true", help="run mode Development")
    parser.add_argument("-e", "--eval", action="store_true", help="run mode Evaluation")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2021 task 2 baseline\nversion {}".format(__VERSIONS__))
        print("===============================\n")
    if args.dev:
        flag = True
    elif args.eval:
        flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag


def load_yaml(yaml_file):
    """
    Load yaml file.
    """
    with open(yaml_file) as stream:
        param = yaml.safe_load(stream)
    return param


def makedir(path):
    os.makedirs(path, exist_ok=True)

    
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_dirs(config, mode):
    """
    Get directory paths according to mode.

    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        print("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=config["dev_directory"]))
    else:
        print("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=config["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


# used in 01_test.py
def get_section_names(target_dir, dir_name, ext="wav"):
    """
    Get section name (almost equivalent to machine ID).

    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath(
        "{target_dir}/{dir_name}/*.{ext}".format(
            target_dir=target_dir, dir_name=dir_name, ext=ext
        )
    )
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(
        list(
            set(
                itertools.chain.from_iterable(
                    [re.findall("section_[0-9][0-9]", ext_id) for ext_id in file_paths]
                )
            )
        )
    )
    return section_names


def file_list_generator(
    target_dir,
    section_name,
    dir_name,
    mode,
    ext="wav",
):
    """
    Get list of audio file paths

    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    print("target_dir : %s" % (target_dir + "_" + section_name))

    prefix_normal = "normal"
    prefix_anomaly = "anomaly"

    # development
    if mode:
        query = os.path.abspath(
            # example:
            # dev_data/ToyCar/train/section_00_source_train_normal_0025_A1Spd28VMic1.wav
            "{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(
                target_dir=target_dir,
                dir_name=dir_name,
                section_name=section_name,
                prefix_normal=prefix_normal,
                ext=ext,
            )
        )
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath(
            "{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(
                target_dir=target_dir,
                dir_name=dir_name,
                section_name=section_name,
                prefix_normal=prefix_anomaly,
                ext=ext,
            )
        )
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        print("number of files : %s" % (str(len(files))))
        if files.size == 0:
            print("no_wav_file!!")

    # evaluation
    else:
        query = os.path.abspath(
            "{target_dir}/{dir_name}/{section_name}_*.{ext}".format(
                target_dir=target_dir,
                dir_name=dir_name,
                section_name=section_name,
                ext=ext,
            )
        )
        files = sorted(glob.glob(query))
        labels = None
        print("number of files : %s" % (str(len(files))))
        if files.size == 0:
            print("no_wav_file!!")
        # print("\n=========================================")

    return files, labels


def save_csv(save_file_path, save_data):
    """
    Save results (AUCs and pAUCs) into csv file.
    """
    with open(save_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(save_data)
