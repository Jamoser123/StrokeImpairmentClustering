#funcs.py
import torch
import numpy as np
import random
import pytorch_lightning as pl
import math
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

def getMLFeatures(cfg):
    feature_columns = [
    'wrist_aff__acc_x', 'wrist_aff__acc_y', 'wrist_aff__acc_z',
    'wrist_aff__gyro_x', 'wrist_aff__gyro_y', 'wrist_aff__gyro_z',
    'wrist_nonaff__acc_x', 'wrist_nonaff__acc_y', 'wrist_nonaff__acc_z',
    'wrist_nonaff__gyro_x', 'wrist_nonaff__gyro_y', 'wrist_nonaff__gyro_z'
    ] if cfg.ml.useGyro else [
    'wrist_aff__acc_x', 'wrist_aff__acc_y', 'wrist_aff__acc_z',
    'wrist_nonaff__acc_x', 'wrist_nonaff__acc_y', 'wrist_nonaff__acc_z',
    ]

    return feature_columns

def print_cfg(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
def create_version_name(cfg, run_type="hyperparam_search"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Main experiment grouping
    experiment_group = f"w{cfg.window_size}_th{cfg.threshold_ratio}/{run_type}"
    
    # Parameter groups
    model_config = f"h{cfg.ml.hidden_dim:03d}_p{cfg.ml.projection_dim:03d}_l{cfg.ml.num_layers:02d}"
    train_config = f"lr{cfg.ml.learning_rate:.1e}_b{cfg.ml.batch_size:03d}"
    optim_config = f"wd{cfg.ml.weight_decay:.1e}_dr{cfg.ml.dropout:.2f}_t{cfg.ml.temperature:.2f}"
    augm_config =  f"noise{cfg.ml.noise_level:.2f}_mask{cfg.ml.mask_ratio:.2f}"
    
    # Combine for version name
    version = f"{experiment_group}/{model_config}/{train_config}/{optim_config}/{augm_config}/{timestamp}"
    
    return version

def parse_time(time_str, reference='s'):
    units = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    if reference=='m':
        units = {
            's': 1/60,
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080
        }
    
    # Handle numeric input
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    # Handle string input
    number = float(''.join(filter(str.isdigit, str(time_str))))
    unit = ''.join(filter(str.isalpha, str(time_str))).lower()
    
    return number * units.get(unit, 0)
    
def set_all_seeds(seed: int = 42):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

    # TODO: Maybe we should move this but Leave it here for now
    all_cpus = set(range(os.cpu_count()))
    # Set affinity to use all CPUs
    os.sched_setaffinity(0, all_cpus)
    
def select_random_patients(percentage):
    """
    Randomly select a percentage of patients from the RUxxx format (1-93).
    
    Args:
        percentage (float): Percentage of patients to select (between 0 and 1)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        list: List of selected patient IDs in RUxxx format
    """
    if not 0 < percentage <= 0.3:
        raise ValueError("Percentage must be between 0 and 0.3")
    
    # Create list of valid patient numbers (1-93)
    valid_numbers = range(1, 94)
    
    # Calculate number of patients to select
    k = math.floor(len(valid_numbers) * percentage)
    
    # Randomly select k numbers
    selected_numbers = random.sample(valid_numbers, k)
    remaining_numbers = [n for n in valid_numbers if n not in selected_numbers]
    
    # Convert both lists to RUxxx format
    selected_patients = [f'RU{str(num).zfill(3)}' for num in selected_numbers]
    remaining_patients = [f'RU{str(num).zfill(3)}' for num in remaining_numbers]

    return sorted(selected_patients), sorted(remaining_patients)

def split_patients(cfg):
    """Split patients into train and validation sets"""
    val_pats, train_pats = ([], []) if cfg.ml.exclude_p_pats == 0 else select_random_patients(cfg.ml.exclude_p_pats)
    #print('No patients ecluded in validation set' if cfg.ml.exclude_p_pats == 0 else f'Following patients in validation set {val_pats}')
    return train_pats, val_pats