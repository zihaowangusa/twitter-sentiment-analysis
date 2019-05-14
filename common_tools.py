# -*- coding: utf-8 -*-
# Author: Zihao Wang

import zipfile


def unzip(zip_filepath, dest_path):
    """
        Unzip
    """
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(path=dest_path)


def get_dataset_filename(zip_filepath):
    """
        Get the Database File Name
    """
    with zipfile.ZipFile(zip_filepath) as zf:
        return zf.namelist()[0]


def cal_acc(true_labels, pred_labels):
    """
        Calculate Accuracy
    """
    n_total = len(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]

    acc = sum(correct_list) / n_total
    return acc
