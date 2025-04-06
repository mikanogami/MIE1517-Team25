'''
Model_dagger0
CW Cumulative CTE: 876.4440673713698
timesteps: 3462, expert_timesteps: 512
CCW Cumulative CTE: 1142.8016374197052
timesteps: 3345, expert_timesteps: 515

Model_dagger1
CW Cumulative CTE: 770.8276746860329
timesteps: 3522, expert_timesteps: 0
CCW Cumulative CTE: 1025.8597603614526
timesteps: 3527, expert_timesteps: 5

Model_dagger2
CW Cumulative CTE: 757.6815053488382
timesteps: 3522, expert_timesteps: 0
CCW Cumulative CTE: 925.0224176236867
timesteps: 3524, expert_timesteps: 0
'''
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import SensorDataset, UniformBatchSampler, collate_fn

def plot_dagger_cte():
    cte_CW = [876.4440673713698, 770.8276746860329, 757.6815053488382]
    pct_expert_CW = [1, 512.0 / 3462.0, 0.0, 0.0]
    cte_CCW = [1142.8016374197052, 1025.8597603614526, 925.0224176236867]
    pct_expert_CCW = [1, 515.0 / 3345.0, 5.0 / 3527.0, 0.0]

    dagger_itr = ['No DAgger', '1 iteration of DAgger', '2 iterations of DAgger']

    plt.plot(dagger_itr, cte_CW, marker='o', linestyle='-')
    plt.title('Cumulative Cross-Track Error for Robot Moving Clockwise')
    plt.ylabel('Cumulative Cross-Track Error (in/min)')
    plt.show()

    plt.plot(dagger_itr, cte_CCW, marker='o', linestyle='-')
    plt.title('Cumulative Cross-Track Error for Robot Moving Counterclockwise')
    plt.ylabel('Cumulative Cross-Track Error (in/min)')
    plt.show()

def plot_training_err():
    epochs = range(1, 501)

    data_dagger0 = np.loadtxt('training_error_0dagger.csv', delimiter=",")
    plt.plot(epochs, data_dagger0, marker='o', linestyle='-')
    plt.title('Training Error During DAgger Iteration 0')
    plt.ylabel('Training Error')
    plt.xlabel('Epochs')
    plt.show()

    data_dagger1 = np.loadtxt('training_error_1dagger.csv', delimiter=",")
    plt.plot(epochs, data_dagger1, marker='o', linestyle='-')
    plt.title('Training Error During DAgger Iteration 1')
    plt.ylabel('Training Error')
    plt.xlabel('Epochs')
    plt.show()

    data_dagger2 = np.loadtxt('training_error_2dagger.csv', delimiter=",")
    plt.plot(epochs, data_dagger2, marker='o', linestyle='-')
    plt.title('Training Error During DAgger Iteration 2')
    plt.ylabel('Training Error')
    plt.xlabel('Epochs')
    plt.show()

def plot_data_dist():
    n_classes = 16
    n_sensors = 5
    batch_size = 256

    files = ['training_data_0_CCW.csv', 'training_data_0_CW.csv', 'training_data_1_CCW.csv', 'training_data_1_CW.csv', 'training_data_2_CCW.csv', 'training_data_2_CW.csv']
    train_dataset = SensorDataset(root_dir="data/projectmodel", num_classes=n_classes, files=files)
    train_sampler = UniformBatchSampler(train_dataset.data, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=collate_fn)
    all_cmds = []

    for i_batch, batch in enumerate(train_loader):
        data, steering_angles = batch
        all_cmds.append(steering_angles)

    # Step 2: Concatenate all the tensors
    all_cmds = torch.cat(all_cmds)

    # Step 3: Count occurrences per class
    cmd_counts = torch.bincount(all_cmds, minlength=n_classes)
    cmd_counts[7] = 0
    # Step 4: Plot histogram
    plt.figure(figsize=(8, 5))
    plt.bar(range(n_classes), cmd_counts.tolist(), tick_label=[str(i) for i in range(n_classes)])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.ylim(0, 800)
    plt.title('Steering Angle Distribution: After 2 DAgger Iterations')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    #plot_training_err()
    #plot_dagger_cte()
    plot_data_dist()