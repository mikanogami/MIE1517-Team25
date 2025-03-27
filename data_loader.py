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
        data_from_file = [np.loadtxt(f, delimiter=",") for f in self.filepaths]
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

'''
class UniformBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset (CustomDataset): Dataset object with categorical labels.
            batch_size (int): Number of samples per batch.
            drop_last (bool): Whether to drop the last batch if it's incomplete.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Find indices for each class
        self.class_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Remove empty classes
        self.class_indices = {k: v for k, v in self.class_indices.items() if len(v) > 0}

        # Convert to numpy arrays for easy sampling
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])  # Shuffle within each class
            self.class_indices[label] = np.array(self.class_indices[label])

        # Compute class distribution
        self.class_counts = Counter(dataset.labels)  # Count occurrences of each class
        self.class_counts = {k: v for k, v in self.class_counts.items() if k in self.class_indices}  # Remove empty classes
        total_samples = sum(self.class_counts.values())

        if total_samples == 0:
            raise ValueError("Dataset contains no valid samples.")

        # Compute relative class frequencies
        self.class_ratios = {label: count / total_samples for label, count in self.class_counts.items()}  # Relative frequencies

        # Compute how many samples per class per batch
        self.samples_per_class = {label: max(1, int(self.class_ratios[label] * batch_size)) for label in self.class_counts}

        # Adjust batch size in case of rounding errors
        remaining = batch_size - sum(self.samples_per_class.values())
        sorted_classes = sorted(self.class_counts, key=lambda x: -self.class_ratios[x])  # Prioritize largest classes
        for i in range(remaining):
            self.samples_per_class[sorted_classes[i % len(sorted_classes)]] += 1  # Distribute extra samples

    def __iter__(self):
        """
        Yield batches of indices with approximately the same class distribution.
        """
        # Shuffle indices within each category before sampling
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])

        # Compute the max number of full batches possible
        min_class_size = min(len(indices) for indices in self.class_indices.values()) if self.class_indices else 0
        min_samples_per_class = min(self.samples_per_class.values()) if self.samples_per_class else 1
        num_batches = min_class_size // min_samples_per_class

        all_batches = []
        for i in range(num_batches):
            batch = []
            for label, num_samples in self.samples_per_class.items():
                start_idx = i * num_samples
                end_idx = start_idx + num_samples
                batch.extend(self.class_indices[label][start_idx:end_idx])
            
            np.random.shuffle(batch)  # Shuffle within batch
            all_batches.append(batch)

        remaining_samples = []
        for label in self.class_indices:
            remaining_samples.extend(self.class_indices[label][num_batches * self.samples_per_class[label]:])
        np.random.shuffle(remaining_samples)

        if len(remaining_samples) >= self.batch_size:
            all_batches.append(remaining_samples[:self.batch_size])

        np.random.shuffle(all_batches)  # Shuffle batch order

        for batch in all_batches:
            yield batch

    def __len__(self):
        """
        Return the number of batches.
        """
        min_class_size = min(len(indices) for indices in self.class_indices.values()) if self.class_indices else 0
        min_samples_per_class = min(self.samples_per_class.values()) if self.samples_per_class else 1
        return (min_class_size // min_samples_per_class) + 1
'''