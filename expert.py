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

#class Expert:
    '''This class represents the expert that navigates the robot successfully through maze without collisions'''
    