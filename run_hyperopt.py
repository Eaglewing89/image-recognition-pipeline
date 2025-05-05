#!/usr/bin/env python
import sys

# Config setup
import config
from functions.utility import analyze_webdataset
import mlflow
import torch
import glob
import os

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config.DEVICE = device

# Configure MLflow
mlflow.set_experiment("animals10")

# Define constants
DATA_DIR = "./data/webdataset/"

# Relative file paths
config.TRAIN_PATHS = sorted(glob.glob(os.path.join(DATA_DIR, "train-*.tar")))
config.TEST_PATHS = sorted(glob.glob(os.path.join(DATA_DIR, "test-*.tar")))

print(f"Found {len(config.TRAIN_PATHS)} training files and {len(config.TEST_PATHS)} test files")

num_classes, class_names, class_weights = analyze_webdataset(DATA_DIR, verbose=False)
print(f"\nTraining data summary:")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Class weights tensor shape: {class_weights.shape}")

# Update the config module variables
config.NUM_CLASSES = num_classes
config.CLASS_NAMES = class_names
config.CLASS_WEIGHTS = class_weights

if __name__ == "__main__":
    db_path = "optuna_animals10_kfold.db"
    from functions.hyperopt import run_kfold_optuna_optimization
    study = run_kfold_optuna_optimization(
        n_trials=2,
        k=3,
        verbose=False,
        storage=db_path,
        load_if_exists=True,
        first_fold_min_acc=95.0
    )
    print(f"Best params: {study.best_params}")