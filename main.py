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
import datetime

sys.path.append(os.path.abspath("simmer-python"))
from maze import Maze
from robot import Robot
from block import Block
from interface.hud import Hud
from interface.communication import TCPServer
import config as CONFIG
import utilities

from expert import Expert

def run_sim():
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
    EXPERT = Expert()   #     TRACK_PID = EXPERT.PID(Kp=1.5, Ki=0.1, Kd=7.5, dt=0.5)
    TRACK_PID = EXPERT.PID(Kp=1, Ki=0, Kd=7.5, dt=0.5)
    TRAJECTORY_PID = EXPERT.PID(Kp=0.03, Ki=1, Kd=5, dt=0.5)

    # Initialize CSV training data list
    training_data = []

    # Initialize graphics
    pygame.init()
    canvas = pygame.display.set_mode([CANVAS_WIDTH, CANVAS_HEIGHT])

    ### Main Loop ###
    RUNNING = True
    expert_bool = True
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

            # This moves robot forward at constant speed and rotates the robot left or right depending on keypress
            # TODO: once expert is implemented
            WALL_DISTANCE = 2.8   # 2

            if True in keypress:
                ROBOT.move_constant_speed_manual([*BLOCK.block_square, *MAZE.reduced_walls], keypress)
            # if u1 < 8 and u2 < 4:  # Obstacle detected in front - 6
            #     print("Wall ahead! Turning left...")
            #     ROBOT.move_constant_speed(walls=[*BLOCK.block_square, *MAZE.reduced_walls], steering_angle=200)
                # print("detect")
            # elif u1 < 8:  # Obstacle detected in front - 6
            #     print("Wall ahead! Turning left...")
            #     ROBOT.move_constant_speed(walls=[*BLOCK.block_square, *MAZE.reduced_walls], steering_angle=5)
            else:
                # Use both track and trajectory error to adjust steering angle
                # ROBOT.move_constant_speed(walls=[*BLOCK.block_square, *MAZE.reduced_walls], steering_angle=steering_adjustment)
                cte, heading_error = EXPERT.get_cte(ROBOT.position, ROBOT.rotation)
                track_adjustment = TRACK_PID.compute(cte)
                trajectory_adjustment = TRAJECTORY_PID.compute(heading_error)
                steering_adjustment = track_adjustment + trajectory_adjustment
                ROBOT.move_constant_speed(walls=[*BLOCK.block_square, *MAZE.reduced_walls], steering_angle=steering_adjustment)


            # Log training data for expert control: timestamp, sensor readings, and steering adjustment
            timestamp = datetime.datetime.now().isoformat()
            training_data.append((timestamp, u0, u1, u2, steering_adjustment))

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

            # Example trajectory (list of (x, y) positions)
            track_width = 12
            maze_dim_x = 96
            maze_dim_y = 48

            tp = {
                "topleft": [0 + track_width/2, 0 + track_width/2],
                "topright": [maze_dim_x - track_width/2, 0 + track_width/2],
                "bottomright": [maze_dim_x - track_width/2, maze_dim_y - track_width/2],
                "bottomleft": [0 + track_width/2, maze_dim_y - track_width/2]
            }

            # Convert inches to pixels for each coordinate in a dictionary
            def inches_to_pixels_dict(inches_dict):
                # Convert each (x, y) coordinate in the dictionary to pixels
                return {key: [value[0] * CONFIG.ppi + CONFIG.border_pixels, value[1] * CONFIG.ppi + CONFIG.border_pixels] for key, value in inches_dict.items()}

            tp_inches = inches_to_pixels_dict(tp)

            # List of trajectory points based on the turning points
            trajectory = [
                tp_inches["topleft"],
                tp_inches["topright"],
                tp_inches["bottomright"],
                tp_inches["bottomleft"],
                tp_inches["topleft"]  # Closing the loop by returning to topleft
            ]


            # Draw the block
            BLOCK.draw(canvas)

            MAZE.draw_trajectory(canvas, trajectory)


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

    except KeyboardInterrupt:
        pass
        
    # Save training data on exit
    with open("expert_training_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "u0", "u1", "u2", "steering_angle"])
        writer.writerows(training_data)
        print('Saved CSV!')

if __name__ == "__main__":
    run_sim()
    
    print('Execution finished. Closing SimMeR.')
    pygame.quit()
