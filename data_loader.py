import os
import math, random
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, Sampler

# function used by dataloader
def collate_fn(batch):
    inputs, labels = batch
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    return inputs, labels

class SensorDataset(Dataset):
    def __init__(self, root_dir, num_classes):
        """
        Args:
            root_dir (str): Directory containing CSV files.
            num_classes (int): Number of categories to discretize the continuous labels.
        """
        self.root_dir = root_dir
        self.n_classes = num_classes
        self.filepaths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".csv")]

        # Load and concatenate all CSV files
        data_from_file = [np.loadtxt(f, delimiter=",") for f in self.filepaths if os.path.getsize(f) > 0]
        self.data = np.vstack(data_from_file)

        # Extract features (sensor values) and labels
        self.features = self.data[:, :-1]   # All columns except last
        self.labels_raw = self.data[:, -1]      # Last column (continuous label)
        self.labels = np.empty(self.labels_raw.shape)

        for idx, label in enumerate(self.labels_raw):
            self.labels[idx] = int(((label + 1.5) / 3.0) * (self.n_classes - 1))
            if self.labels[idx] < 0:
                print(f"bad label: {self.labels[idx]}, raw label: {self.labels_raw[idx]}, features: {self.features[idx]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

'''
UniformBatchSampler code was adapted from an assignment for another course at UofT.
Course: CSC2626 Introduction to Imitation Learning
Instructor: Florian Shkurti
Availability: https://github.com/florianshkurti/csc2626w22/blob/master/assignments/A1/dataset_loader.py
'''
class UniformBatchSampler(Sampler):

    def __init__(self, data, batch_size, classes=-1):
        # build data for sampling here
        self.labels = data[:, -1]      # Last column (continuous label)
        self.batch_size = batch_size
        self.classes = classes
        self.num_batch = math.floor(len(self.labels) / self.batch_size)
        self.cmds_ordered = []

        # [{key: label, value: list of idx of images with given label}, ...]
        cmds_dict = {}
        # [images with label=1, images with label=2, ...]
        # each set of images with same label are shuffled before added to list
        cmds_shuffled = []

        # self.cmds_dict is a dict where key: label and value: list of idx that label occurs
        for idx, label in enumerate(self.labels):
            steering_command = np.array(label, dtype=np.float32)
            steering_command = int(((steering_command + 1.5)/3.0) * (self.classes - 1)) 

            if steering_command in cmds_dict:
                cmds_dict.get(steering_command).append(idx)
            else:
                cmds_dict[steering_command] = [idx]

        # our "filler" data will be the data with the most common label
        filler_key = max(cmds_dict, key = lambda x: len(cmds_dict.get(x)))
        filler_data = cmds_dict.get(filler_key)
        cmds_dict.pop(filler_key)

        for key in cmds_dict:
            data = cmds_dict.get(key)
            random.shuffle(data)
            cmds_shuffled.extend(data)

        random.shuffle(filler_data)
        cmds_shuffled.extend(filler_data)

        cmds_batched = {}

        for i in range(0, self.batch_size):
            for i_batch in range(0, self.num_batch):
                next_item = cmds_shuffled[0]
                del cmds_shuffled[0]

                if i_batch in cmds_batched:
                    cmds_batched.get(i_batch).append(next_item)
                else:
                    cmds_batched[i_batch] = [next_item]
        
        for i_batch in range(0, self.num_batch):
            self.cmds_ordered.extend(cmds_batched.get(i_batch))

    def __iter__(self):
        # implement logic of sampling here
        batch = []

        for i, cmd in enumerate(self.cmds_ordered):
            batch.append(cmd)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []


    def __len__(self):
        return len(self.labels)