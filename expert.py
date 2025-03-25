# Imports
import numpy as np
import pygame
import sys, os

sys.path.append(os.path.abspath("simmer-python"))
from maze import Maze
from robot import Robot
from block import Block
from interface.hud import Hud
from interface.communication import TCPServer
import config as CONFIG
import utilities

class Expert:
    '''This class represents the expert that navigates the robot successfully through maze implemented using PD controller'''
    #def simulate(self, u0, u1, u2):
    def __init__(self, Kp, Ki, Kd, dt, da=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0
        self.integral = 0
        self.filtered_derivative = 0
        self.derivative_alpha = da

    def compute(self, error):
        self.integral += error * self.dt

        raw_derivative = (error - self.prev_error) / self.dt
        # Apply low-pass filter to the derivative term
        self.filtered_derivative = (
            self.derivative_alpha * raw_derivative +
            (1 - self.derivative_alpha) * self.filtered_derivative
        )

        output = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * self.filtered_derivative
        )

        self.prev_error = error
        return output