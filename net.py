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

def train_model(model, train_loader, test_loader):
    print("training model")

net = RobotControlNet()
x_in = torch.randn(100, 3)
angle, move = net(x_in)

print(angle.shape, move.shape)