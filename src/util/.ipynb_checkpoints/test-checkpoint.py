#test.py
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_simple_time_series(data, title):
    plt.figure(figsize=(12, 4))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label='Feature {}'.format(i))
    plt.title(title)
    #plt.legend()
    plt.show()

def plot_time_series(data, title):
    """
    Plot time series data with separate subplots for each feature.
    
    Args:
        data: numpy array or torch tensor of shape [seq_len, features]
        title: string for the overall plot title
    """
    # Convert to numpy if tensor
    if torch.is_tensor(data):
        data = data.numpy()
    
    # Define feature names
    feature_names = [
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(6, 2, figsize=(15, 20))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Colors for affected and non-affected sides
    affected_color = 'blue'
    nonaffected_color = 'red'
    
    # Plot each feature
    for i in range(6):  # For each feature type (acc_x, acc_y, etc.)
        # Affected side
        axes[i, 0].plot(data[:, i], color=affected_color)
        axes[i, 0].set_title(f'Affected {feature_names[i]}')
        axes[i, 0].grid(True)
        
        # Non-affected side
        axes[i, 1].plot(data[:, i+6], color=nonaffected_color)
        axes[i, 1].set_title(f'Non-affected {feature_names[i]}')
        axes[i, 1].grid(True)
        
        # Add y-labels only to leftmost plots
        if i < 3:  # Accelerometer
            axes[i, 0].set_ylabel('Acceleration (m/s²)')
        else:  # Gyroscope
            axes[i, 0].set_ylabel('Angular velocity (rad/s)')
    
    # Add x-labels to bottom plots
    for ax in axes[-1,:]:
        ax.set_xlabel('Time steps')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()