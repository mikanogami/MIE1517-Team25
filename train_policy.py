import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
from net import RobotControlNet
from data_loader import SensorDataset, UniformBatchSampler, collate_fn

def train_model(model, train_loader, dagger_itr, save_path, learning_rate=1e-3, num_epochs=100):
    print("Start training model")
    # Fixed PyTorch random seed for reproducible results
    torch.manual_seed(25)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

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

            #y_pred = F.softmax(outputs, 1)

            # weight errors according to inverse frequency of occurance (prevent overfitting from data that is unequally distributed across classes)
            num_bins = model.n_classes
            cmd_counts = torch.bincount(labels.flatten(), minlength=num_bins)
            inv_weights = torch.empty(num_bins)
            for cmd_idx, count in enumerate(cmd_counts):
                if count == 0:
                    inv_weights[cmd_idx] = 1
                else:
                    inv_weights[cmd_idx] = 1/count

            loss_fn = nn.CrossEntropyLoss(weight=inv_weights)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            total_train_err += (predicted != labels).sum().item()  # Count errors
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)

        print(f"Epoch {epoch + 1}: Train err: {train_err[epoch]:.4f}, Train loss: {train_loss[epoch]:.4f}")    

    model_path = (f"{save_path}/model_dagger{dagger_itr}_lr{learning_rate}_ep{epoch}")
    torch.save(model.state_dict(), model_path)
    print("Finished training")

def plot_data_hist():
    print("test")

def main():
    n_classes = 16
    n_sensors = 5
    batch_size = 256

    train_dataset = SensorDataset(root_dir="data", num_classes=n_classes)
    train_sampler = UniformBatchSampler(train_dataset.data, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=collate_fn)

    # test that UniformBatchSampler samples batches with approximately the same distribution of classes across batches
    for i_batch, batch in enumerate(train_loader):
        data, steering_angles = batch
        cmd_counts = torch.bincount(steering_angles, minlength=n_classes)
        print('Batch {} has distribution: {}'.format(i_batch, cmd_counts))
    

    model = RobotControlNet(ultrasonic_dim=n_sensors, n_classes=n_classes)
    train_model(model, train_loader, dagger_itr=2, num_epochs=500)

    

if __name__ == "__main__":
    main()