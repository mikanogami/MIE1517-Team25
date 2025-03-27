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
import csv
import datetime, time
import torch

sys.path.append(os.path.abspath("simmer-python"))
from maze import Maze
from robot import Robot
from block import Block
from interface.hud import Hud
from interface.communication import TCPServer
import config as CONFIG
import utilities

from expert import Expert
from net import RobotControlNet

def run_sim(model=None, save_expert_data=False, dagger_itr=None, runtime=None):
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

    # Create expert that controls robot
    EXPERT = Expert()
    TRACK_PID = EXPERT.PID(Kp=2, Ki=3, Kd=7.5, dt=0.5)
    TRAJECTORY_PID = EXPERT.PID(Kp=0.0001, Ki=0.5, Kd=1, dt=0.5)

    # Initialize CSV training data list
    training_data = []

    # Initialize graphics
    pygame.init()
    canvas = pygame.display.set_mode([CANVAS_WIDTH, CANVAS_HEIGHT])

    ### Main Loop ###
    RUNNING = True
    expert_bool = True if model is None else False
    start_time = time.time()

    try:
        while RUNNING:

            ##########################
            ##### USER INTERFACE #####
            ##########################
            # Check for keyboard input
            game_events = pygame.event.get()
            RUNNING = HUD.check_input(game_events)
            keypress = pygame.key.get_pressed()

            # Manually simulate a specific sensor or sensors
            #utilities.simulate_sensors(environment, SIMULATE_LIST)

            u0 = ROBOT.sensors.get('u0').simulate(0, environment)   # left sensor reading
            u1 = ROBOT.sensors.get('u1').simulate(0, environment)   # middle sensor reading
            u2 = ROBOT.sensors.get('u2').simulate(0, environment)   # right sensor reading

            if expert_bool:
                # Use both track and trajectory error to adjust steering angle
                cte, heading_error = EXPERT.get_cte(ROBOT.position, ROBOT.rotation, CW=CONFIG.clockwise)
                track_adjustment = TRACK_PID.compute(cte)
                trajectory_adjustment = TRAJECTORY_PID.compute(heading_error)
                steering_adjustment =  track_adjustment + trajectory_adjustment
                # set bound on steering adjustment (required for our classification model)
                if steering_adjustment > 1.5:
                    steering_adjustment = 1.5
                elif steering_adjustment < -1.5:
                    steering_adjustment = -1.5
                ROBOT.move_constant_speed(walls=[*BLOCK.block_square, *MAZE.reduced_walls], steering_angle=steering_adjustment)
                if save_expert_data:
                    training_data.append((u0, u1, u2, steering_adjustment))

            else:
                sensor_vals = np.array([u0, u1, u2])
                steering_adjustment = model.get_steering_cmd(sensor_vals)
                ROBOT.move_constant_speed(walls=[*BLOCK.block_square, *MAZE.reduced_walls], steering_angle=steering_adjustment)

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
            HUD.draw_expert_indicator(canvas, expert_bool)

            # Limit the framerate
            HUD.clock.tick(CONFIG.frame_rate)

            # Flip the display (update the canvas)
            pygame.display.flip()

            # check that we have not surpassed runtime
            if time.time() - start_time > runtime:
                RUNNING = False

    except KeyboardInterrupt:
        pass
    
    print(time.time() - start_time)
    # Save training data on exit
    if save_expert_data:
        with open(f"data_tmp/training_data_{dagger_itr}_{'CW' if CONFIG.clockwise else 'CCW'}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(training_data)
            print('Saved CSV!')

if __name__ == "__main__":
    n_classes = 20

    agent = RobotControlNet(n_classes=n_classes)
    agent.load_state_dict(torch.load("Model_lr0.001_ep299"))
    
    # run simulation with robot moving clockwise
    start_pos_CW = [6, 36]
    start_rot_CW = 180
    CONFIG.robot_start_position = start_pos_CW
    CONFIG.robot_start_rotation = start_rot_CW
    CONFIG.clockwise = True
    run_sim(model=agent, save_expert_data=False, dagger_itr=1, runtime=120)

    # run simulation with robot moving counter clockwise
    start_pos_CCW = [6, 12]
    start_rot_CCW = 0
    CONFIG.robot_start_position = start_pos_CCW
    CONFIG.robot_start_rotation = start_rot_CCW
    CONFIG.clockwise = False
    run_sim(model=agent, save_expert_data=False, dagger_itr=1, runtime=120)

    print('Execution finished. Closing SimMeR.')
    pygame.quit()
