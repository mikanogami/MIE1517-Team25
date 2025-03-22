'''
This is the main file of SimMeR.
'''
# This file is part of SimMeR, an educational mechatronics robotics simulator.
# Initial development funded by the University of Toronto MIE Department.
# Copyright (C) 2023  Ian G. Bennett
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

from expert import Expert

### Initialization
print('SimMeR Loading...')

# Set random error seed
if not CONFIG.rand_error:
    np.random.seed(CONFIG.floor_seed)

# Load maze walls and floor pattern
MAZE = Maze()
MAZE.import_walls()
MAZE.generate_floor()
CANVAS_WIDTH = MAZE.size_x * CONFIG.ppi + CONFIG.border_pixels * 2
CANVAS_HEIGHT = MAZE.size_y * CONFIG.ppi + CONFIG.border_pixels * 2

# Load robot
ROBOT = Robot()

# List of sensors to simulate every frame (for testing only)
if hasattr(CONFIG, 'simulate_list'):
    SIMULATE_LIST = CONFIG.simulate_list
else:
    SIMULATE_LIST = []

# Create the block
BLOCK = Block()

# Create a copy of the environment objects to pass to simulation functions
environment = {'BLOCK': BLOCK, 'MAZE': MAZE, 'ROBOT': ROBOT}

# Load the Heads Up Display
HUD = Hud()

# Load TCP Communication
COMM = TCPServer()
COMM.start()

# Create expert that controls robot
EXPERT = Expert(kp=0.25, kd=0.025)

#speed = 

# Initialize graphics
pygame.init()
canvas = pygame.display.set_mode([CANVAS_WIDTH, CANVAS_HEIGHT])

### Main Loop ###
RUNNING = True
expert_control = True
try:
    while RUNNING:
        
        ##########################
        ##### USER INTERFACE #####
        ##########################
        # Check for keyboard input
        game_events = pygame.event.get()
        RUNNING = HUD.check_input(game_events)
        keypress = pygame.key.get_pressed()

        # Get the command information from the tcp buffer
        cmds = COMM.get_buffer_rx()

        ################################################
        ##### ROBOT AND DEVICE UPDATES AND ACTIONS #####
        ################################################
        # Act on commands and respond
        if cmds:
            responses = ROBOT.command(cmds, environment)
            COMM.set_buffer_tx(responses)

        # Manually simulate a specific sensor or sensors
        #utilities.simulate_sensors(environment, SIMULATE_LIST)

        u0 = ROBOT.sensors.get('u0').simulate(0, environment)   # left sensor reading
        u1 = ROBOT.sensors.get('u1').simulate(0, environment)   # middle sensor reading
        u2 = ROBOT.sensors.get('u2').simulate(0, environment)   # right sensor reading

        # TODO: EXPERT takes robot position and orientation and outputs steering angle
        # TODO: anytime EXPERT is controlling robot, it records sensor data (u0, u1, u2) and associated steering angle (this is our training data)
        # TODO: NET takes u0, u1, u2 sensor readings and outputs steering angle

        # this moves robot forward at constant speed and rotates the robot left or right depending on keypress
        # TODO: once expert is implemented
        if True in keypress:
            ROBOT.move_constant_speed_manual([*BLOCK.block_square, *MAZE.reduced_walls], keypress)
        else:
            ROBOT.move_constant_speed_manual([*BLOCK.block_square, *MAZE.reduced_walls])

        # Recalculate global positions of the robot and its devices
        ROBOT.update_outline()
        ROBOT.update_device_positions()

        ###########################################
        ##### DRAW RELEVANT OBJECTS ON CANVAS #####
        ###########################################
        # Fill the background with the background color
        canvas.fill(CONFIG.background_color)

        # Draw the maze checkerboard pattern
        MAZE.draw_floor(canvas)

        # Draw the maze walls
        MAZE.draw_walls(canvas)

        # Draw the block
        BLOCK.draw(canvas)

        # Draw the robot onto the maze
        ROBOT.draw(canvas)
        ROBOT.draw_devices(canvas)

        # Update the various HUD elements
        HUD.draw_frame_indicator(canvas)
        HUD.draw_keys(canvas, keypress)

        # Limit the framerate
        HUD.clock.tick(CONFIG.frame_rate)

        # Flip the display (update the canvas)
        pygame.display.flip()

except KeyboardInterrupt:
    pass

# Done! Time to quit.
print('Execution finished. Closing SimMeR.')
pygame.quit()
