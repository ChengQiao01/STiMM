# -*- coding: utf-8 -*-
"""
Load training data set

Created on Feb 2025

@author: Cheng Qiao (chengqiao21@mails.jlu.edu.cn)
"""
import numpy as np
import h5py
import scipy
import torch


def DataLoad_Train(train_data_dir, dataname, data_type, chanell_dim=None, expand_dim=None):
    print(train_data_dir, dataname)
    # Load .mat data
    if data_type == 'v7':
        data1_set = h5py.File(train_data_dir, 'r')
        data1_set = np.float32(data1_set[str(dataname)])
    elif data_type == 'v5':
        data1_set = scipy.io.loadmat(train_data_dir)
        data1_set = np.float32(data1_set[str(dataname)])
    else:
        raise ValueError("Unsupported data_type. Use 'v7' or 'v5'.")

    # Transpose if chanell_dim is provided
    if chanell_dim is not None:
        data1_set = np.transpose(data1_set, chanell_dim)

    # Expand dimensions if expand_dim is provided
    if expand_dim is not None:
        train_set = np.expand_dims(data1_set, axis=expand_dim)
    else:
        train_set = data1_set
    train_set = torch.from_numpy(train_set)
    return train_set


def generate_random_bool_masked_pos(batch_size, num_patches, mask_ratio=0.5):
    num_masked = int(mask_ratio * num_patches)

    bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool)

    for i in range(batch_size):
        mask_indices = torch.randperm(num_patches)[:num_masked]
        bool_masked_pos[i, mask_indices] = True

    return bool_masked_pos

