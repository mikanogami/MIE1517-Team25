# Imports
import numpy as np
import pygame
import sys, os, math

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

    def get_cte(self, position, rot, CW=True):
        '''
        Calculates cross track error for robot
        Inputs: 
        - position, rot are the current x and y position and orientation of the robot
        - CW (bool): true if robot is moving CW, false for CCW
        Assumes: track width = 12, maze is 96 by 48
        Returns: 
        - cte: cross track error (perp. distance between robot and reference trajectory)
        - heading_error: difference between robot orientation and reference trajectory tangent direction
        '''
        x, y = position

        track_width = 12
        maze_dim_x = 96
        maze_dim_y = 48
        # turning points on rectangular track
        tp = {
            "topleft": [0 + track_width, 0 + track_width], 
            "topright": [maze_dim_x - track_width, 0 + track_width], 
            "bottomright": [maze_dim_x - track_width, maze_dim_y - track_width], 
            "bottomleft": [0 + track_width, maze_dim_y - track_width]}

        # straight section: vertical track on the left
        if 0 <= x <= track_width and track_width <= y <= maze_dim_y - track_width:
            cte = abs(x - track_width) - track_width/2
            heading = 180 if CW else 0
        # turning section: top-left
        elif 0 <= x < track_width and 0 <= y < track_width:
            x_c, y_c = tp["topleft"]
            cte = math.dist([x_c,y_c], [x,y]) - track_width/2
            theta_ref = math.atan2(y-y_c, x-x_c)    # angle of vector between turning point and robot position
            heading = theta_ref + 90 if CW else theta_ref - 90
        # straight section: horizontal track on the top
        elif track_width <= x <= maze_dim_x - track_width and 0 <= y <= track_width:
            cte = abs(y - track_width) - track_width/2
            heading = 270 if CW else 90
        # turning section: top-right
        elif maze_dim_x - track_width < x <= maze_dim_x and 0 <= y < track_width:
            x_c, y_c = tp["topright"]
            cte = math.dist([x_c,y_c], [x,y]) - track_width/2
            theta_ref = math.atan2(y-y_c, x-x_c)   
            heading = theta_ref + 90 if CW else theta_ref - 90
        # straight section: vertical track on the right
        elif maze_dim_x - track_width <= x <= maze_dim_x and track_width <= y <= maze_dim_y:
            cte = abs(x - (maze_dim_x - track_width)) - track_width/2
            heading = 0 if CW else 180
        # turning section: bottom-right
        elif maze_dim_y - track_width < x <= maze_dim_x and maze_dim_y - track_width < y <= maze_dim_y:
            x_c, y_c = tp["bottomright"]
            cte = math.dist([x_c,y_c], [x,y]) - track_width/2
            theta_ref = math.atan2(y-y_c, x-x_c)   
            heading = theta_ref + 90 if CW else theta_ref - 90
        # straight section: horizontal track on the bottom
        elif track_width <= x <= maze_dim_y - track_width and maze_dim_y - track_width <= y <= maze_dim_y:
            cte = abs(y - (maze_dim_y - track_width)) - track_width/2
            heading = 90 if CW else 270
        # turning section: bottom-left
        elif 0 <= x < track_width and maze_dim_y - track_width < y <= maze_dim_y: 
            x_c, y_c = tp["bottomleft"]
            cte = math.dist([x_c,y_c], [x,y]) - track_width/2
            theta_ref = math.atan2(y-y_c, x-x_c)   
            heading = theta_ref + 90 if CW else theta_ref - 90
        # ERROR: robot is outside of track
        else:
            raise Exception("Robot outside of track")

        heading_error = (rot - heading + 180) % 360 - 180
        return cte, heading_error
        
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