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

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
from net import RobotControlNet
from data_loader import SensorDataset, UniformBatchSampler, collate_fn
from train_policy import train_model
from main import run_sim
from pathlib import Path

def get_model_name(args):
    return f"{args['id']}_{args['n_sensors']}sensors_{args['n_classes']}classes_{args['deploy_time']}runtime_{args['n_dagger']}dagger"

def run_CW_track(model, dagger_itr, save_dir, runtime=120):
    # run simulation with robot moving clockwise
    start_pos_CW = [6, 36]
    start_rot_CW = 180
    CONFIG.robot_start_position = start_pos_CW
    CONFIG.robot_start_rotation = start_rot_CW
    CONFIG.clockwise = True
    run_sim(model=model, dagger_itr=dagger_itr, runtime=runtime, save_data_dir=save_dir)

def run_CCW_track(model, dagger_itr, save_dir, runtime=120):
    # run simulation with robot moving counter clockwise
    start_pos_CCW = [6, 12]
    start_rot_CCW = 0
    CONFIG.robot_start_position = start_pos_CCW
    CONFIG.robot_start_rotation = start_rot_CCW
    CONFIG.clockwise = False
    run_sim(model=model, dagger_itr=dagger_itr, runtime=runtime, save_data_dir=save_dir)

def run_dagger(args):
    data_save_dir = f'data/{get_model_name(args)}'
    models_save_dir = f'models/{get_model_name(args)}'

    agent = None
    for i_dagger in range(args['n_dagger']):
        run_CW_track(model=agent, dagger_itr=i_dagger, save_dir=data_save_dir, runtime=args['deploy_time'])
        run_CCW_track(model=agent, dagger_itr=i_dagger, save_dir=data_save_dir, runtime=args['deploy_time'])
        pygame.quit()

        train_dataset = SensorDataset(root_dir=f"data/{get_model_name(args)}", num_classes=args['n_classes'])
        train_sampler = UniformBatchSampler(train_dataset.data, batch_size=args['batch_size'])
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=collate_fn)

        agent = RobotControlNet(ultrasonic_dim=args['n_sensors'], n_classes=args['n_classes'])
        train_model(agent, train_loader, dagger_itr=i_dagger, save_path=models_save_dir, num_epochs=args['n_epochs'])

    run_CW_track(model=agent, dagger_itr=i_dagger, save_dir=None, runtime=args['deploy_time'])
    run_CCW_track(model=agent, dagger_itr=i_dagger, save_dir=None, runtime=args['deploy_time'])


if __name__ == "__main__":
    args = {
        'id': 'default', 
        'n_classes': 16, 
        'n_sensors': 5, 
        'batch_size': 256,
        'n_dagger': 6,
        'n_epochs': 300,
        'deploy_time': 120 # time we run CW and CCW simulation to collect data (seconds)
    }

    # create subfolder in data/ directory under unique model name
    data_path = Path(f'data/{get_model_name(args)}')
    if data_path.exists() and data_path.is_dir() and any(data_path.iterdir()):
        raise RuntimeError(f"Folder {data_path} exists and is not empty!")
    else:
        data_path.mkdir(parents=True, exist_ok=True)
    # create subfolder in models/ directory under unique model name
    model_path = Path(f'models/{get_model_name(args)}')
    if model_path.exists() and model_path.is_dir() and any(model_path.iterdir()):
        raise RuntimeError(f"Folder {model_path} exists and is not empty!")
    else:
        model_path.mkdir(parents=True, exist_ok=True)

    run_dagger(args)