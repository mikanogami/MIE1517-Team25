import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os

from data_loader import SensorDataset, UniformBatchSampler

def train_model(model, train_loader, learning_rate=0.001, num_epochs=30):
    print("Start training model")
    # Fixed PyTorch random seed for reproducible results
    torch.manual_seed(25)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            total_train_err += (predicted != labels).sum().item()  # Count errors
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)

        print(f"Epoch {epoch + 1}: Train err: {train_err[epoch]:.4f}, Train loss: {train_loss[epoch]:.4f}")    
        model_path = (f"Model_lr{learning_rate}_ep{epoch}")
        torch.save(net.state_dict(), model_path)

    print("Finished training")


def main():
    n_classes = 15
    batch_size = 256
    train_dataset = SensorDataset(root_dir="data", num_classes=n_classes)
    train_sampler = UniformBatchSampler(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, sampler=train_sampler)

    # test UniformBatchSampler
    
    for i_batch, batch in enumerate(train_loader):
        data, steering_angles = batch
        cmd_counts = torch.bincount(steering_angles.flatten(), minlength=n_classes)
        print('Batch {} has distribution: {}'.format(i_batch, cmd_counts))
    

if __name__ == "__main__":
    main()