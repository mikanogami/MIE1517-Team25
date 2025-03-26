import argparse
import math
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.misc
import torch
import torch.nn as nn
import torch.optim as optim


class RobotControlNet(nn.Module):
    def __init__(self, ultrasonic_dim=3, n_classes=20):
        super().__init__()

        self.shared_layer = nn.Sequential(
            nn.Linear(ultrasonic_dim, 32),
            nn.LeakyReLU(0.1),
        )

        self.steering_angle = nn.Linear(32, n_classes)
        self.do_move = nn.Linear(32, 2)

    def forward(self, x):
        x = self.shared_layer(x)
        angle = self.steering_angle(x)
        move = self.do_move(x)

        return angle, move

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

 
net = RobotControlNet()
x_in = torch.randn(100, 3)
angle, move = net(x_in)

print(angle.shape, move.shape)