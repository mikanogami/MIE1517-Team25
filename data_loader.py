import os
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, Sampler

class SensorDataset(Dataset):
    def __init__(self, root_dir, num_classes):
        """
        Args:
            root_dir (str): Directory containing CSV files.
            num_classes (int): Number of categories to discretize the continuous labels.
        """
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.filepaths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".csv")]

        # Load and concatenate all CSV files
        self.data = pd.concat([pd.read_csv(f, header=None) for f in self.filepaths], ignore_index=True)

        # Extract features (sensor values) and labels
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)  # All columns except last
        self.labels = self.data.iloc[:, -1].values.astype(np.float32)  # Last column (continuous label)

        # Convert continuous labels into categorical bins
        self.labels = self._discretize_labels(self.labels)

    def _discretize_labels(self, labels):
        """Discretizes continuous labels into 'num_classes' bins."""
        min_label, max_label = labels.min(), labels.max()
        bins = np.linspace(min_label, max_label, self.num_classes + 1)  # Create num_classes bins
        categorical_labels = np.digitize(labels, bins, right=True) - 1  # Convert to bin indices

        # Ensure labels are within valid range [0, num_classes-1]
        categorical_labels = np.clip(categorical_labels, 0, self.num_classes - 1)
        return categorical_labels.astype(np.int64)  # Convert to PyTorch-compatible dtype

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Long for classification tasks
        return x, y


class UniformBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        """
        Args:
            dataset (CustomDataset): Dataset object with categorical labels.
            batch_size (int): Number of samples per batch.
            drop_last (bool): Whether to drop the last batch if it's incomplete.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
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

        # Handle remaining samples if drop_last is False
        if not self.drop_last:
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
        return min_class_size // min_samples_per_class if self.drop_last else (min_class_size // min_samples_per_class) + 1
