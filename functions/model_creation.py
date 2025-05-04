# Import essential libraries
import os
import sys
import tempfile
import shutil
import random
import numpy as np
import torch
import config
from config import TRAIN_PATHS, TEST_PATHS, NUM_CLASSES, CLASS_NAMES, CLASS_WEIGHTS, device, SEED
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import mlflow
import mlflow.pytorch
import optuna
from optuna.samplers import TPESampler
import webdataset as wds
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display
import seaborn as sns
import glob
import io
from PIL import Image
import collections
from tqdm.auto import tqdm
from scipy import stats

# Define a function to monitor memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
        print(f"GPU memory: Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")

def analyze_webdataset(path_pattern, verbose=True):
    """
    Analyze a WebDataset to extract class information and compute class weights.
    
    Args:
        path_pattern (str): Path pattern for the WebDataset shards
        verbose (bool): Whether to print analysis information
    
    Returns:
        tuple: (num_classes, class_names, class_weights)
            - num_classes (int): Number of unique classes in the dataset
            - class_names (list): List of class names
            - class_weights (torch.Tensor): Tensor of class weights for balanced training
    """
    if verbose:
        print(f"Analyzing WebDataset at {path_pattern}...")
    
    # If path_pattern is a directory, list all shard files
    if os.path.isdir(path_pattern):
        shard_files = sorted([
            os.path.join(path_pattern, f) 
            for f in os.listdir(path_pattern) 
            if f.startswith('train-') and f.endswith('.tar')
        ])
        if verbose:
            print(f"Found {len(shard_files)} shard files in directory")
    else:
        # If it's a pattern or single file
        if os.path.exists(path_pattern):
            shard_files = [path_pattern]
        else:
            import glob
            shard_files = sorted(glob.glob(path_pattern))
            if verbose:
                print(f"Found {len(shard_files)} shard files matching pattern")
    
    if not shard_files:
        raise ValueError(f"No WebDataset shards found at {path_pattern}")
    
    # Process all shards to count class frequencies
    class_counts = collections.Counter()
    total_samples = 0
    
    for shard in shard_files:
        try:
            dataset = wds.WebDataset(shard)
            
            for sample in tqdm(dataset, desc=f"Processing {os.path.basename(shard)}", 
                              disable=not verbose):
                try:
                    # Ensure we have a 'cls' field
                    if 'cls' in sample:
                        # Decode class name if it's in bytes format
                        if isinstance(sample['cls'], bytes):
                            class_name = sample['cls'].decode('utf-8')
                        else:
                            class_name = sample['cls']
                        
                        class_counts[class_name] += 1
                        total_samples += 1
                except Exception as e:
                    if verbose:
                        print(f"Error processing sample: {e}")
                    continue
        except Exception as e:
            if verbose:
                print(f"Error processing shard {shard}: {e}")
            continue
    
    # Get class names sorted by frequency (most common first)
    class_names = [class_name for class_name, _ in class_counts.most_common()]
    num_classes = len(class_names)
    
    if verbose:
        print(f"\nFound {num_classes} unique classes in {total_samples} samples")
        print("\nClass distribution:")
        for class_name, count in class_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
    
    # Calculate class weights for balanced training (inverse of frequency)
    # Using the formula: n_samples / (n_classes * class_freq)
    class_weights = []
    for class_name in class_names:
        class_freq = class_counts[class_name]
        weight = total_samples / (num_classes * class_freq)
        class_weights.append(weight)
    
    # Convert to PyTorch tensor
    weights_tensor = torch.FloatTensor(class_weights)
    
    if verbose:
        print("\nClass weights:")
        for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
            print(f"  {class_name}: {weight:.4f}")
    
    return num_classes, class_names, weights_tensor


# Define data augmentation pipelines with different intensities
def get_transforms(intensity='medium'):
    # Define normalization for ResNet50
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Define transforms based on intensity
    if intensity == 'low':
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    elif intensity == 'medium':
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    elif intensity == 'high':
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError(f"Unknown intensity: {intensity}")
    
    # Validation transform is always the same (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform
# Function to create data loaders with WebDataset
def create_dataloaders(batch_size=32, augmentation_intensity='medium', val_split=0.1, num_workers=4):
    # Get transforms based on augmentation intensity
    train_transform, val_transform = get_transforms(augmentation_intensity)
    
    # Define preprocessing function for WebDataset
    def preprocess_train(sample):
        # Convert image bytes to PIL Image and apply transform
        image = Image.open(io.BytesIO(sample['jpg']))
        tensor_image = train_transform(image)
        
        # Get class name and convert to index
        class_name = sample['cls'].decode('utf-8')
        class_idx = CLASS_NAMES.index(class_name)
        
        return tensor_image, class_idx
    
    def preprocess_val(sample):
        # Convert image bytes to PIL Image and apply transform
        image = Image.open(io.BytesIO(sample['jpg']))
        tensor_image = val_transform(image)
        
        # Get class name and convert to index
        class_name = sample['cls'].decode('utf-8')
        class_idx = CLASS_NAMES.index(class_name)
        
        return tensor_image, class_idx
        
    # Split TRAIN_PATHS into training and validation sets
    num_val_shards = max(1, int(len(TRAIN_PATHS) * val_split))
    # Ensure deterministic split with the random seed
    rng = random.Random(SEED)
    all_paths = TRAIN_PATHS.copy()
    rng.shuffle(all_paths)
    
    train_shard_paths = all_paths[num_val_shards:]
    val_shard_paths = all_paths[:num_val_shards]
    
    print(f"Split {len(TRAIN_PATHS)} shards into {len(train_shard_paths)} training and {len(val_shard_paths)} validation shards")
    
    # Create training dataset with WebDataset
    train_dataset = (
        wds.WebDataset(train_shard_paths, shardshuffle=True)
        .map(preprocess_train)
        .shuffle(1000)
    )
    
    # Calculate appropriate number of workers for test set
    test_workers = min(len(TEST_PATHS), num_workers)
    if test_workers == 0:
        test_workers = 1  # Ensure at least one worker
    
    # Create validation dataset with a separate subset of training data
    # No need for shuffling or sharp shuffle in validation
    val_dataset = (
        wds.WebDataset(val_shard_paths, shardshuffle=False)
        .map(preprocess_val)
    )
    
    # Create test dataset
    test_dataset = (
        wds.WebDataset(TEST_PATHS, shardshuffle=False, resampled=False)
        .map(preprocess_val)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=test_workers)
    
    print(f"Created loaders with {num_workers} train workers and {test_workers} test workers")
    
    return train_loader, val_loader, test_loader

# Calculate class weights for handling imbalance (approximate)
def compute_class_weights(dataloader):
    # This is an approximation for WebDataset
    # In a real implementation, you might want to pre-compute this
    class_counts = torch.zeros(NUM_CLASSES)
    max_samples = 1000  # Limit to 1000 samples for quick estimation
    
    for i, (_, labels) in enumerate(tqdm(dataloader, desc="Computing class weights")):
        if i * dataloader.batch_size >= max_samples:
            break
            
        for label in labels:
            class_counts[label] += 1
    
    # If a class has 0 samples, set its count to 1 to avoid division by zero
    class_counts = torch.max(class_counts, torch.ones_like(class_counts))
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * NUM_CLASSES
    return weights
# Define the model architecture with ResNet50 backbone
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2, freeze_backbone=True):
        super(AnimalClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionally unfreeze the last convolutional block
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original FC layer
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# Function to create model, loss function, and optimizer
def create_model(learning_rate, dropout_rate, weight_decay, class_weights=None):
    model = AnimalClassifier(NUM_CLASSES, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Create weighted loss function if class weights are provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=weight_decay)
    
    return model, criterion, optimizer
# Training function
def train_epoch(model, train_loader, criterion, optimizer, scheduler=None, disable_progress=False, batch_size=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(train_loader, desc="Training", disable=disable_progress)
    
    # Limit batches during training to avoid exhausting dataset
    max_batches = int(1024 / batch_size)  # 1024 samples per epoch
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Stop after max_batches
        if batch_idx >= max_batches:
            break
            
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
    
    # Calculate epoch statistics (only for the batches we processed)
    if total > 0:  # Avoid division by zero
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
    else:
        epoch_loss = 0
        epoch_acc = 0
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, val_loader, criterion, disable_progress=False, batch_size=8):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Limit batches for evaluation to avoid exhausting dataset
    max_batches = int(512 / batch_size)  # 512 samples for evaluation
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Evaluating", disable=disable_progress)):
            # Stop after max_batches
            if batch_idx >= max_batches:
                break
                
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for later analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate statistics
    if total > 0:  # Avoid division by zero
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
    else:
        epoch_loss = 0
        epoch_acc = 0
        all_preds = []
        all_labels = []
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Full training loop with early stopping
def train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=20, patience=5, verbose=True, batch_size=8):
    # Initialize the learning rate scheduler (OneCycleLR)
    # WebDataset doesn't support len(), so we hardcode the steps_per_epoch
    steps_per_epoch = int(1024/batch_size)  # We're limiting to 100 batches per epoch in train_epoch anyway
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        anneal_strategy='cos'
    )
    
    # Initialize tracking variables
    best_val_acc = 0.0
    best_model_state = None
    no_improve_epochs = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Clear GPU cache before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print_gpu_memory()
    
    # Training loop
    for epoch in range(max_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{max_epochs}")
            print("-" * 20)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, disable_progress=not verbose, batch_size=batch_size)
        
        # Clear intermediate tensors to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, disable_progress=not verbose, batch_size=batch_size)
        
        # Print results if verbose mode is enabled
        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print_gpu_memory()
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Log metrics with MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early stopping check
        if no_improve_epochs >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc
# Utility functions for managing Optuna studies
def list_optuna_studies(storage_path):
    """List all Optuna studies in the given SQLite database"""
    storage_string = f"sqlite:///{storage_path}"
    try:
        studies = optuna.study.get_all_study_summaries(storage=storage_string)
        if len(studies) > 0:
            print(f"Found {len(studies)} studies in {storage_path}:")
            for i, study in enumerate(studies):
                print(f"  {i+1}. {study.study_name} - Direction: {study.direction.name}, " 
                      f"Trials: {study.n_trials}, Best value: {study.best_trial.value if study.best_trial else 'None'}")
            return studies
        else:
            print(f"No studies found in {storage_path}")
            return []
    except Exception as e:
        print(f"Error accessing studies in {storage_path}: {e}")
        return []

def load_optuna_study(storage_path, study_name):
    """Load a specific Optuna study from a SQLite database"""
    storage_string = f"sqlite:///{storage_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_string)
        print(f"Loaded study '{study_name}' with {len(study.trials)} trials")
        
        if study.best_trial:
            print(f"Best value: {study.best_value} with parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
        else:
            print("No completed trials with valid values found.")
            
        return study
    except Exception as e:
        print(f"Error loading study '{study_name}' from {storage_path}: {e}")
        return None

def analyze_optuna_study(study):
    """Analyze an Optuna study and display useful visualizations"""
    if study is None or len(study.trials) == 0:
        print("No valid study or trials available for analysis.")
        return
    
    # Basic statistics
    print(f"Study name: {study.study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Get trials dataframe
    trials_df = study.trials_dataframe()
    display(trials_df.head())
    
    # Plot optimization history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    
    # Parameter importance if there are enough trials
    if len(study.trials) >= 2:
        plt.subplot(1, 2, 2)
        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title('Parameter Importance')
        except Exception as e:
            print(f"Could not plot parameter importance: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Plot parallel coordinate plot for parameters
    if len(study.trials) >= 2:
        try:
            plt.figure(figsize=(12, 6))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.title('Parallel Coordinate Plot')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot parallel coordinates: {e}")
    
    return trials_df
def create_stratified_kfolds(shard_paths, k=5, seed=SEED, verbose=True):
    """
    Create stratified k-folds from WebDataset shards maintaining class distribution.
    
    Args:
        shard_paths: List of paths to WebDataset tar files
        k: Number of folds
        seed: Random seed for reproducibility
        
    Returns:
        list: List of k lists, where each inner list contains shard paths for one fold
    """
    # Analyze class distribution in each shard
    shard_class_distributions = {}
    if verbose:
        print(f"Analyzing class distributions in {len(shard_paths)} shards...")
    
    for shard_path in shard_paths:
        try:
            dataset = wds.WebDataset(shard_path)
            class_counts = collections.Counter()
            
            for sample in dataset:
                if 'cls' in sample:
                    class_name = sample['cls'].decode('utf-8') if isinstance(sample['cls'], bytes) else sample['cls']
                    class_counts[class_name] += 1
            
            # Store class distribution for this shard
            shard_class_distributions[shard_path] = class_counts
        except Exception as e:
            print(f"Error analyzing shard {os.path.basename(shard_path)}: {e}")
            # If there's an error, use an empty counter
            shard_class_distributions[shard_path] = collections.Counter()
    
    # Create a signature for each shard based on its class distribution
    # This will help us distribute shards to maintain class balance
    shard_signatures = {}
    for shard_path, counts in shard_class_distributions.items():
        # Normalize the counts to get proportions
        total = sum(counts.values()) if sum(counts.values()) > 0 else 1
        signature = {cls: count/total for cls, count in counts.items()}
        shard_signatures[shard_path] = signature
    
    # Initialize k empty folds
    folds = [[] for _ in range(k)]
    
    # Initialize counter for class counts per fold
    fold_class_counts = [collections.Counter() for _ in range(k)]
    
    # Sort shards by total sample count (descending)
    sorted_shards = sorted(shard_signatures.keys(), 
                           key=lambda s: sum(shard_class_distributions[s].values()),
                           reverse=True)
    
    # Use consistent randomization
    rng = random.Random(seed)
    rng.shuffle(sorted_shards)
    
    # Assign shards to folds using a greedy approach
    for shard in sorted_shards:
        # Find the fold with the lowest total count
        fold_idx = min(range(k), key=lambda i: sum(fold_class_counts[i].values()))
        
        # Add shard to this fold
        folds[fold_idx].append(shard)
        
        # Update the class counts for this fold
        for cls, count in shard_class_distributions[shard].items():
            fold_class_counts[fold_idx][cls] += count
    
    if verbose:
        # Print fold statistics
        print(f"Created {k} folds with the following statistics:")
        for i, (fold, counts) in enumerate(zip(folds, fold_class_counts)):
            total = sum(counts.values())
            print(f"Fold {i+1}: {len(fold)} shards, {total} samples")
            
            # Print class distribution in this fold
            if total > 0:
                for cls in CLASS_NAMES:
                    percent = (counts[cls] / total) * 100 if cls in counts else 0
                    print(f"  {cls}: {counts[cls]} samples ({percent:.2f}%)")
    
    return folds

def objective_kfold(trial, k=3, verbose=True, first_fold_min_acc=90.0):
    """
    Objective function with k-fold cross validation for Optuna optimization.
    Implements a hybrid pruning approach:
    1. Early pruning based on first fold performance
    2. Epoch-level pruning within folds
    3. Progressive fold evaluation
    
    Args:
        trial: Optuna trial object
        k: Number of folds for cross-validation
        verbose: Whether to print verbose output
        
    Returns:
        float: t-statistic lower bound of validation accuracy
    """
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    augmentation_intensity = trial.suggest_categorical("augmentation_intensity", ["low", "medium", "high"])
    patience = trial.suggest_int("patience", 3, 10)
    max_epochs = trial.suggest_int("max_epochs", 10, 30)
    
    # Force smaller batch size for GPU memory constraints
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 8.0:
            batch_size = min(batch_size, 32)
        torch.cuda.empty_cache()
    
    # Create k-fold split of the training shards
    folds = create_stratified_kfolds(TRAIN_PATHS, k=k, seed=SEED, verbose=verbose)
    
    # Start MLflow run for this trial
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}_kfold"):
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "augmentation_intensity": augmentation_intensity,
            "patience": patience,
            "max_epochs": max_epochs,
            "k_folds": k
        })
        
        # Set threshold for first-fold pruning
        # This should be set based on domain knowledge; here using a conservative value
        first_fold_min_acc = first_fold_min_acc  # Minimum acceptable accuracy for first fold (%)
        
        # Track validation accuracies for each fold
        fold_accuracies = []
        epoch_accuracies = {}  # Dictionary to store accuracies by epoch
        
        # For each fold
        for fold_idx in range(k):
            if verbose:
                print(f"\n--- Fold {fold_idx+1}/{k} ---")
            
            # Define train and validation shards for this fold
            val_shard_paths = folds[fold_idx]
            train_shard_paths = []
            for i in range(k):
                if i != fold_idx:
                    train_shard_paths.extend(folds[i])
            
            if verbose:
                print(f"Training on {len(train_shard_paths)} shards, validating on {len(val_shard_paths)} shards")
            
            # Get transforms
            train_transform, val_transform = get_transforms(augmentation_intensity)
            
            # Define preprocessing functions
            def preprocess_train(sample):
                image = Image.open(io.BytesIO(sample['jpg']))
                tensor_image = train_transform(image)
                class_name = sample['cls'].decode('utf-8')
                class_idx = CLASS_NAMES.index(class_name)
                return tensor_image, class_idx
            
            def preprocess_val(sample):
                image = Image.open(io.BytesIO(sample['jpg']))
                tensor_image = val_transform(image)
                class_name = sample['cls'].decode('utf-8')
                class_idx = CLASS_NAMES.index(class_name)
                return tensor_image, class_idx
            
            # Create datasets
            train_dataset = (
                wds.WebDataset(train_shard_paths, shardshuffle=True)
                .map(preprocess_train)
                .shuffle(1000)
            )
            
            val_dataset = (
                wds.WebDataset(val_shard_paths, shardshuffle=False)
                .map(preprocess_val)
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)
            
            # Create model with the specified hyperparameters
            model, criterion, optimizer = create_model(
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                class_weights=CLASS_WEIGHTS
            )
            
            try:
                # For the first fold, implement custom training with per-epoch pruning checks
                if fold_idx == 0:
                    # Implement a custom training loop that allows epoch-level pruning for first fold
                    scheduler = optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=optimizer.param_groups[0]['lr'],
                        steps_per_epoch=int(1024/batch_size),  # Same as in train_model
                        epochs=max_epochs,
                        anneal_strategy='cos'
                    )
                    
                    # Initialize tracking variables
                    best_val_acc = 0.0
                    best_model_state = None
                    no_improve_epochs = 0
                    fold_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
                    
                    # Training loop with pruning checks
                    for epoch in range(max_epochs):
                        if verbose:
                            print(f"Epoch {epoch+1}/{max_epochs}")
                        
                        # Train for one epoch
                        train_loss, train_acc = train_epoch(
                            model, train_loader, criterion, optimizer, scheduler, 
                            disable_progress=not verbose,
                            batch_size=batch_size
                        )
                        
                        # Clear intermediate tensors to free up memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Evaluate on validation set
                        val_loss, val_acc, _, _ = evaluate(
                            model, val_loader, criterion, 
                            disable_progress=not verbose,
                            batch_size=batch_size
                        )
                        
                        # Print results if verbose
                        if verbose:
                            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                        
                        # Update history
                        fold_history["train_loss"].append(train_loss)
                        fold_history["train_acc"].append(train_acc)
                        fold_history["val_loss"].append(val_loss)
                        fold_history["val_acc"].append(val_acc)
                        
                        # Store the validation accuracy for this epoch
                        if epoch not in epoch_accuracies:
                            epoch_accuracies[epoch] = []
                        epoch_accuracies[epoch].append(val_acc)
                        
                        # Report to Optuna for pruning
                        trial.report(val_acc, epoch)
                        
                        # Check if this trial should be pruned based on current epoch performance
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned(
                                f"Trial pruned at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%"
                            )
                        
                        # Early stopping check
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model_state = model.state_dict().copy()
                            no_improve_epochs = 0
                        else:
                            no_improve_epochs += 1
                        
                        if no_improve_epochs >= patience:
                            if verbose:
                                print(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    
                    # After first fold, check if performance is promising
                    if best_val_acc < first_fold_min_acc:
                        if verbose:
                            print(f"First fold accuracy ({best_val_acc:.2f}%) below threshold ({first_fold_min_acc:.2f}%). Pruning trial.")
                        
                        # Log that we pruned based on first fold
                        mlflow.log_param("pruned_first_fold", True)
                        mlflow.log_metric("first_fold_acc", best_val_acc)
                        
                        # Raise pruned exception
                        raise optuna.exceptions.TrialPruned(
                            f"Trial pruned after first fold with best val acc: {best_val_acc:.2f}%"
                        )
                    
                    # Add the best accuracy from first fold
                    fold_accuracies.append(best_val_acc)
                    
                # For subsequent folds, use the standard train_model function
                else:
                    # Train the model normally for subsequent folds
                    model, history, best_val_acc = train_model(
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        max_epochs=max_epochs,
                        patience=patience,
                        verbose=verbose and fold_idx == 0,  # Only show verbose output for first fold
                        batch_size=batch_size
                    )
                    
                    # Store the validation accuracy for this fold
                    fold_accuracies.append(best_val_acc)

                    # Store validation accuracies by epoch
                    for epoch, val_acc in enumerate(history['val_acc']):
                        if epoch not in epoch_accuracies:
                            epoch_accuracies[epoch] = []
                        epoch_accuracies[epoch].append(val_acc)
                
                # After each fold, check if the trial is promising compared to others
                if fold_idx > 0:  # Skip check for first fold (already handled)
                    # Calculate average accuracy across completed folds
                    avg_acc_so_far = sum(fold_accuracies) / len(fold_accuracies)
                    
                    # Log the current average
                    mlflow.log_metric(f"avg_acc_after_{fold_idx+1}_folds", avg_acc_so_far)
                    
                    # Get median of completed trials so far (if we have enough)
                    completed_trials = [t for t in trial.study.trials 
                                      if t.state == optuna.trial.TrialState.COMPLETE]
                    
                    if len(completed_trials) >= 5:  # Need at least 5 completed trials for meaningful comparison
                        values = [t.value for t in completed_trials if t.value is not None]
                        if values:  # Make sure we have valid values
                            median_value = np.median(values)
                            
                            # If we're significantly below median, prune
                            if avg_acc_so_far < (median_value * 0.85):  # Threshold at 85% of median
                                if verbose:
                                    print(f"Performance ({avg_acc_so_far:.2f}%) below 85% of median ({median_value:.2f}%). Pruning.")
                                
                                # Log pruning decision
                                mlflow.log_param("pruned_progressive", True)
                                mlflow.log_param("pruned_after_fold", fold_idx + 1)
                                mlflow.log_metric("avg_acc_at_pruning", avg_acc_so_far)
                                mlflow.log_metric("median_value_at_pruning", median_value)
                                
                                # Raise pruned exception
                                raise optuna.exceptions.TrialPruned(
                                    f"Trial pruned after fold {fold_idx+1}/{k} with avg_acc={avg_acc_so_far:.2f}% (below median)"
                                )
                
                # Clear GPU memory after each fold
                del model
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                # Handle CUDA memory errors
                if "CUDA" in str(e) and ("out of memory" in str(e) or "unknown error" in str(e)):
                    if verbose:
                        print(f"CUDA memory error in fold {fold_idx+1}")
                    torch.cuda.empty_cache()
                    return float('-inf')
                else:
                    raise e
                
        # Calculate metrics using the epoch-wise approach
        epoch_avg_accuracies = {}
        epoch_std_accuracies = {}
        
        # Only use epochs where all folds have data
        for epoch, accs in epoch_accuracies.items():
            if len(accs) == k:  # Only consider epochs with data from all k folds
                # Calculate average
                epoch_avg = sum(accs) / k
                epoch_avg_accuracies[epoch] = epoch_avg
                
                # Log with consistent metric name and varying step value
                mlflow.log_metric("epoch_avg_val_acc", epoch_avg, step=epoch)
                
                # Calculate standard deviation
                if len(accs) > 1:  # Need at least 2 values for std
                    epoch_std = np.std(accs)
                    epoch_std_accuracies[epoch] = epoch_std
                    
                    # Log standard deviation as a separate metric with the same step
                    mlflow.log_metric("epoch_std_val_acc", epoch_std, step=epoch)

        # Find best average validation accuracy and its epoch
        if epoch_avg_accuracies:
            best_epoch = max(epoch_avg_accuracies.items(), key=lambda x: x[1])[0]
            best_avg_val_acc = epoch_avg_accuracies[best_epoch]
            
            mlflow.log_metric("best_avg_epoch", best_epoch)
            mlflow.log_metric("best_avg_val_acc", best_avg_val_acc)

            # Log this as the recommended number of epochs for final training
            mlflow.log_param("recommended_epochs", best_epoch + 1)  # +1 because epochs are 0-indexed
            
            # Log the standard deviation of the best epoch if available
            if best_epoch in epoch_std_accuracies:
                best_std_val_acc = epoch_std_accuracies[best_epoch]
                
                # Use 80% confidence with t-distribution for small samples
                confidence_level = 0.80
                t_critical = stats.t.ppf(confidence_level, df=k-1)
                
                # Calculate lower bound
                lower_bound = best_avg_val_acc - (t_critical * best_std_val_acc / np.sqrt(k))
                
                # Log all metrics
                mlflow.log_metric("best_avg_val_acc", best_avg_val_acc)
                mlflow.log_metric("best_std_val_acc", best_std_val_acc)
                mlflow.log_metric("lower_confidence_bound", lower_bound)
                
                print(f"Best avg validation at epoch {best_epoch+1}: {best_avg_val_acc:.2f}% ± {best_std_val_acc:.2f}%")
                print(f"Objective value - t-dist Lower confidence bound ({confidence_level*100}%): {lower_bound:.2f}%")

                # Store MLflow run id in trial user attributes for later reference
                trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
                
                return lower_bound
        else:
            return float('-inf')
        
def run_kfold_optuna_optimization(n_trials=5, k=5, verbose=True, study_name="animals10_kfold", storage=None, load_if_exists=True, first_fold_min_acc=90.0):
    """
    Run Optuna hyperparameter optimization with k-fold cross validation
    
    Args:
        n_trials: Number of Optuna trials to run
        k: Number of folds for cross-validation
        verbose: Whether to print verbose output
        study_name: Name of the Optuna study
        storage: Storage URL for the Optuna study
        load_if_exists: Whether to load existing study if it exists
        
    Returns:
        optuna.Study: The completed Optuna study
    """
    # Prepare storage string for SQLite if provided
    storage_string = None
    if storage:
        storage_string = f"sqlite:///{storage}"
        print(f"Using SQLite storage at: {storage_string}")
    
    # Configure logging for Optuna
    if not verbose:
        import logging
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.setLevel(logging.WARNING)
    
    # Create a fresh study object or load existing one
    study = None
    
    if storage and load_if_exists:
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_string
            )
            
            if len(study.trials) > 0:
                print(f"Loaded existing study with {len(study.trials)} previous trials.")
                print(f"Best value so far: {study.best_value} t-dist 80% lower bound, with parameters:")
                for key, value in study.best_params.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Could not load existing study: {e}")
            study = None
    
    if study is None:
        # Create a pruner - here we use the MedianPruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Number of trials to run before pruning starts
            n_warmup_steps=5,    # Number of epochs to run before pruning can happen in a trial
            interval_steps=1     # Evaluate pruning conditions after each epoch
        )
        
        study = optuna.create_study(
            direction="maximize",  # Maximize validation accuracy
            sampler=TPESampler(seed=SEED),
            pruner=pruner,        # Add the pruner here
            study_name=study_name,
            storage=storage_string,
            load_if_exists=load_if_exists
        )
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print_gpu_memory()
    
    # Define objective function wrapper
    def objective_wrapper(trial):
        return objective_kfold(trial, k=k, verbose=verbose, first_fold_min_acc=first_fold_min_acc)
    
    # Create callback function to show progress
    already_printed_trials = set()
    
    def trial_callback(study, trial):
        if trial.number not in already_printed_trials and trial.state in [
            optuna.trial.TrialState.COMPLETE, 
            optuna.trial.TrialState.PRUNED
        ]:
            already_printed_trials.add(trial.number)
            if not verbose:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    print(f"Trial {trial.number} completed with value: {trial.value:.2f}")
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    print(f"Trial {trial.number} pruned at epoch {trial.last_step}")
    
    # Run optimization
    try:
        study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[trial_callback])
        
        # Print summary
        print("\nK-Fold Study statistics:")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Best trial:")
        trial = study.best_trial
        
        if trial.value is not None:
            print(f"    Value: {trial.value} t-dist 80% lower bound")
            print(f"    Params:")
            for key, value in trial.params.items():
                print(f"      {key}: {value}")
        else:
            print("    No valid value found for best trial.")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    return study
def visualize_classifications(model, test_loader, save_dir="./"):
    """
    Visualize correctly and incorrectly classified images from the test set.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test dataset
        save_dir: Directory to save visualization images
    
    Returns:
        tuple: Paths to the saved visualization images
    """
    model.eval()
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Dictionaries to store correctly classified images by class
    correct_samples = {class_name: [] for class_name in CLASS_NAMES}
    incorrect_samples = []
    
    # Process images and collect samples
    print("Collecting sample images for visualization...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Finding samples"):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Make predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Move data back to CPU for storage
            inputs_cpu = inputs.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            # Check each image in the batch
            for i in range(len(labels_cpu)):
                true_label = labels_cpu[i]
                pred_label = preds_cpu[i]
                true_class = CLASS_NAMES[true_label]
                pred_class = CLASS_NAMES[pred_label]
                
                # If correctly classified, add to the correct class list
                if true_label == pred_label:
                    # Only keep one sample per class
                    if len(correct_samples[true_class]) < 1:
                        correct_samples[true_class].append({
                            'image': inputs_cpu[i],
                            'true_class': true_class,
                            'pred_class': pred_class
                        })
                # If incorrectly classified, add to incorrect list
                else:
                    incorrect_samples.append({
                        'image': inputs_cpu[i],
                        'true_class': true_class,
                        'pred_class': pred_class
                    })
            
            # Check if we have enough samples
            all_correct_filled = all(len(samples) >= 1 for samples in correct_samples.values())
            if all_correct_filled and len(incorrect_samples) >= 10:
                break
    
    # Function to denormalize images for display
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)
    
    # Create figure for correct classifications
    plt.figure(figsize=(15, 6))
    plt.suptitle("Correctly Classified Samples (1 per class)", fontsize=16)
    
    # Create a grid of correct images
    for i, class_name in enumerate(CLASS_NAMES):
        if correct_samples[class_name]:
            sample = correct_samples[class_name][0]
            plt.subplot(2, 5, i + 1)
            
            # Denormalize and convert to displayable format
            img = denormalize(sample['image']).permute(1, 2, 0).numpy()
            
            plt.imshow(img)
            plt.title(f"{class_name}", fontsize=10)
            plt.axis('off')
        else:
            plt.subplot(2, 5, i + 1)
            plt.text(0.5, 0.5, f"No correct\n{class_name}", 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    correct_path = os.path.join(save_dir, "correct_classifications.png")
    plt.savefig(correct_path, dpi=150)
    plt.close()  # Close the figure to free memory
    
    # Create figure for incorrect classifications
    plt.figure(figsize=(15, 6))
    plt.suptitle("Incorrectly Classified Samples", fontsize=16)
    
    # Randomly select incorrect samples if we have more than needed
    if len(incorrect_samples) > 10:
        incorrect_samples = random.sample(incorrect_samples, 10)
    
    # Create a grid of incorrect images
    for i, sample in enumerate(incorrect_samples[:10]):
        plt.subplot(2, 5, i + 1)
        
        # Denormalize and convert to displayable format
        img = denormalize(sample['image']).permute(1, 2, 0).numpy()
        
        plt.imshow(img)
        plt.title(f"True: {sample['true_class']}\nPred: {sample['pred_class']}", fontsize=9)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    incorrect_path = os.path.join(save_dir, "incorrect_classifications.png")
    plt.savefig(incorrect_path, dpi=150)
    plt.close()  # Close the figure to free memory
    
    print(f"Visualization images saved to {correct_path} and {incorrect_path}")
    
    return correct_path, incorrect_path

# Full evaluation function without batch limits
def evaluate_full(model, test_loader, criterion, disable_progress=False):
    """Evaluate model on the full dataset without batch limits"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    print(f"Evaluating model on full test set (no batch limit)...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Full Evaluation", disable=disable_progress):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for later analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Periodically clear GPU cache to avoid memory issues
            if torch.cuda.is_available() and (len(all_preds) % (50 * test_loader.batch_size) == 0):
                torch.cuda.empty_cache()
    
    # Calculate statistics
    if total > 0:  # Avoid division by zero
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
    else:
        epoch_loss = 0
        epoch_acc = 0
        all_preds = []
        all_labels = []
    
    print(f"Evaluated on {total} samples: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Function to train the final model with best hyperparameters
def train_final_model(study, epochs_factor=1.2):
    """Train a final model with the best hyperparameters using all available data.
    Uses early stopping information from the best trial to avoid overfitting.
    """
    if study is None or len(study.trials) == 0:
        print("No valid study available. Cannot train final model.")
        return None, 0
        
    # Get best hyperparameters and trial
    best_trial = study.best_trial
    best_params = best_trial.params
    print("Training final model with best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Determine ideal training duration based on early stopping from best trial
    mlflow_run_id = best_trial.user_attrs.get("mlflow_run_id")
    
    if mlflow_run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(mlflow_run_id)

            final_epochs = int(run.data.params["recommended_epochs"])
            print(f"Using recommended_epochs from k-fold CV: {final_epochs}")
        except Exception as e:
            print(f"Could not retrieve early stopping info: {e}")
            final_epochs = int(best_params["max_epochs"] * 0.8)
            print(f"Using fallback: will train for {final_epochs} epochs")
    else:
        final_epochs = int(best_params["max_epochs"] * 0.8)
        print(f"No run info available. Will train for {final_epochs} epochs")
    
    # Create combined train+val loader
    combined_paths = TRAIN_PATHS
    print(f"Using all {len(combined_paths)} training shards")
    
    # Use the best hyperparameters
    augmentation_intensity = best_params["augmentation_intensity"]
    batch_size = best_params["batch_size"]
    
    # Create dataloaders
    train_transform, _ = get_transforms(augmentation_intensity)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define preprocessing functions
    def preprocess_train(sample):
        image = Image.open(io.BytesIO(sample['jpg']))
        tensor_image = train_transform(image)
        class_name = sample['cls'].decode('utf-8')
        class_idx = CLASS_NAMES.index(class_name)
        return tensor_image, class_idx
    
    def preprocess_test(sample):
        image = Image.open(io.BytesIO(sample['jpg']))
        tensor_image = test_transform(image)
        class_name = sample['cls'].decode('utf-8')
        class_idx = CLASS_NAMES.index(class_name)
        return tensor_image, class_idx
    
    # Create datasets
    full_train_dataset = (
        wds.WebDataset(combined_paths, shardshuffle=True)
        .map(preprocess_train)
        .shuffle(1000)
    )
    
    test_dataset = (
        wds.WebDataset(TEST_PATHS, shardshuffle=False, resampled=False)
        .map(preprocess_test)
    )
    
    # Create data loaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    
    # Create model with best hyperparameters
    model, criterion, optimizer = create_model(
        learning_rate=best_params["learning_rate"],
        dropout_rate=best_params["dropout_rate"],
        weight_decay=best_params["weight_decay"],
        class_weights=CLASS_WEIGHTS
    )
    
    # Create temporary directory for artifacts
    artifact_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for artifacts: {artifact_dir}")
    
    # Start MLflow run for final model
    with mlflow.start_run(run_name="final_model_full_training"):
        mlflow.log_params({
            **best_params,
            "training_type": "full_dataset",
            "early_stopping": False,
            "actual_epochs": final_epochs
        })
        
        # Calculate steps per epoch
        steps_per_epoch = int(16384/batch_size)  # A reasonable default estimate
        
        # Training loop with fixed number of epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=final_epochs,
            anneal_strategy='cos'
        )
        
        # Store best model state
        best_model_state = None
        
        # Training loop
        for epoch in range(final_epochs):
            print(f"\nEpoch {epoch+1}/{final_epochs}")
            print("-" * 20)
            
            # Train for one epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc="Training")
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
                
                # Periodically clear GPU cache
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
            epoch_acc = 100 * correct / total if total > 0 else 0
            
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
            print_gpu_memory()

            # Call scheduler.step() once per epoch
            scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_acc": epoch_acc
            }, step=epoch)
            
            # Save checkpoint to the artifact directory
            checkpoint_path = os.path.join(artifact_dir, 'model_checkpoint.pth')
            torch.save(model.state_dict(), checkpoint_path)
            
            # Only in the last epoch, keep the state for final evaluation
            if epoch == final_epochs - 1:
                best_model_state = model.state_dict().copy()
        
        # Load the best model state if we have one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save the final model to the artifact directory
        model_path = os.path.join(artifact_dir, 'final_model.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "final_model")
        
        # Only evaluate once on the full test set
        print("\nEvaluating final model on test set...")
        test_loss, test_acc, test_preds, test_labels = evaluate_full(model, test_loader, criterion)
        
        print(f"\nFinal Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        
        # Log test metrics
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        # Generate and log per-class metrics
        if len(test_preds) > 0 and len(test_labels) > 0:
            try:
                report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)
                
                # Log per-class metrics
                for class_name in CLASS_NAMES:
                    mlflow.log_metric(f"test_f1_{class_name}", report[class_name]['f1-score'])
                    mlflow.log_metric(f"test_precision_{class_name}", report[class_name]['precision'])
                    mlflow.log_metric(f"test_recall_{class_name}", report[class_name]['recall'])
                
                # Create and log confusion matrix to artifact directory
                cm = confusion_matrix(test_labels, test_preds)
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Final Model Confusion Matrix (Test Set)')
                plt.tight_layout()
                
                cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)

                # Generate classification visualizations
                print("\nGenerating classification visualizations...")
                visualize_classifications(model, test_loader, save_dir=artifact_dir)
                
                # Log all artifacts from the directory
                for artifact_file in os.listdir(artifact_dir):
                    if artifact_file.endswith('.png'):
                        mlflow.log_artifact(os.path.join(artifact_dir, artifact_file))

            except Exception as e:
                print(f"Error generating classification report: {e}")
        
        # Clean up the temporary directory
        try:
            shutil.rmtree(artifact_dir)
            print(f"Cleaned up temporary artifact directory: {artifact_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")
                
    return model, test_acc

def enhanced_optuna_analysis(study, show_plots=True, top_n_trials=10):
    """
    Display comprehensive insights about an Optuna hyperparameter optimization study.
    
    Args:
        study: Optuna study object
        show_plots: Whether to display visualizations
        top_n_trials: Number of top trials to analyze in detail
        
    Returns:
        dict: Dictionary with summary statistics
    """
    if study is None or len(study.trials) == 0:
        print("No valid study or trials available for analysis.")
        return {}
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    # Basic statistics
    stats = {
        "study_name": study.study_name,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "completion_rate": len(completed_trials) / len(study.trials) if study.trials else 0,
    }
    
    print(f"Study: {stats['study_name']}")
    print(f"Total trials: {stats['total_trials']}")
    print(f"Completed trials: {stats['completed_trials']} ({stats['completion_rate']:.1%})")
    print(f"Pruned trials: {stats['pruned_trials']} ({1-stats['completion_rate']:.1%})")
    
    if len(completed_trials) == 0:
        print("No completed trials available for analysis.")
        return stats
    
    # Get trials dataframe
    trials_df = study.trials_dataframe()
    display(trials_df.head())
    
    # Analyze values of completed trials
    values = [t.value for t in completed_trials if t.value is not None]
    if values:
        stats["best_value"] = max(values)
        stats["worst_value"] = min(values)
        stats["median_value"] = np.median(values)
        stats["mean_value"] = np.mean(values)
        stats["std_value"] = np.std(values)
        
        print("\nPerformance statistics (objective value):")
        print(f"  Best: {stats['best_value']:.2f}")
        print(f"  Median: {stats['median_value']:.2f}")
        print(f"  Mean: {stats['mean_value']:.2f} ± {stats['std_value']:.2f}")
        print(f"  Worst completed: {stats['worst_value']:.2f}")
    
    # Top trials analysis
    if completed_trials:
        print(f"\nTop {min(top_n_trials, len(completed_trials))} trials:")
        
        # Sort trials by value (descending)
        sorted_trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
        
        for i, trial in enumerate(sorted_trials[:top_n_trials]):
            print(f"  {i+1}. Trial {trial.number}: {trial.value:.2f}")
            for key, value in trial.params.items():
                print(f"     {key}: {value}")
            print()
    
    # Parameter importance analysis
    if len(completed_trials) >= 2:
        print("\nParameter importance analysis:")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f"  {param}: {importance:.4f}")
            
            # Parameter distributions
            print("\nParameter distributions in successful trials:")
            param_names = list(study.best_params.keys())
            for param in param_names:
                values = [t.params[param] for t in completed_trials if param in t.params]
                if not values:
                    continue
                    
                if isinstance(values[0], (int, float)):
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = min(values)
                    max_val = max(values)
                    best_val = study.best_params[param]
                    
                    print(f"  {param}:")
                    print(f"    Range: {min_val} to {max_val}")
                    print(f"    Mean: {mean_val:.4g} ± {std_val:.4g}")
                    print(f"    Best trial: {best_val}")
                else:
                    # For categorical parameters
                    counter = collections.Counter(values)
                    best_val = study.best_params[param]
                    print(f"  {param}:")
                    for val, count in counter.most_common():
                        marker = " (best)" if val == best_val else ""
                        print(f"    {val}: {count} trials{marker}")
        except Exception as e:
            print(f"Could not analyze parameter importance: {e}")
    
    # Visualizations
    if show_plots and len(completed_trials) >= 2:
        # Optimization history
        plt.figure(figsize=(12, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title('Optimization History')
        plt.tight_layout()
        plt.show()
        
        # Parameter importances
        try:
            plt.figure(figsize=(12, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title('Parameter Importances')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
        
        # Parallel coordinate plot for different parameters
        try:
            plt.figure(figsize=(14, 7))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.title('Parallel Coordinate Plot')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot parallel coordinates: {e}")
            
    
    return stats

def visualize_best_trial_metrics(study, mlflow_client=None):
    """
    Visualize epoch-wise metrics from the best trial in an Optuna study.
    Shows average validation accuracy and standard deviation over epochs.
    
    Args:
        study: Optuna study object
        mlflow_client: MLflow client (optional, will be created if None)
    """
    if study is None or not study.best_trial:
        print("No best trial available for visualization.")
        return
    
    best_trial = study.best_trial
    print(f"Best Trial: #{best_trial.number}")
    print(f"Value: {best_trial.value}")
    print("\nParameters:")
    for param_name, param_value in best_trial.params.items():
        print(f"  {param_name}: {param_value}")
    
    # Get MLflow run ID from trial user attributes
    mlflow_run_id = best_trial.user_attrs.get("mlflow_run_id")
    if not mlflow_run_id:
        print("\nCannot find MLflow run ID in trial user attributes.")
        return
    
    print(f"\nMLflow Run ID: {mlflow_run_id}")
    
    # Create MLflow client if not provided
    if mlflow_client is None:
        mlflow_client = mlflow.tracking.MlflowClient()
    
    # Get run data
    try:
        run = mlflow_client.get_run(mlflow_run_id)
    except Exception as e:
        print(f"Error retrieving MLflow run: {e}")
        return
    
    # Fetch all metrics history
    avg_acc_history = mlflow_client.get_metric_history(mlflow_run_id, "epoch_avg_val_acc")
    std_acc_history = mlflow_client.get_metric_history(mlflow_run_id, "epoch_std_val_acc")
    
    if not avg_acc_history:
        print("No epoch_avg_val_acc metrics found for this run.")
        return
    
    # Extract epoch (step) and values
    epochs = [m.step for m in avg_acc_history]
    avg_values = [m.value for m in avg_acc_history]
    
    std_values = []
    if std_acc_history:
        # Make sure std_values aligns with avg_values epochs
        std_dict = {m.step: m.value for m in std_acc_history}
        std_values = [std_dict.get(epoch, 0) for epoch in epochs]
    
    # Best epoch from params (subtract 1 because epochs are 0-indexed in the code)
    best_avg_epoch = int(run.data.params.get("recommended_epochs", 0)) - 1
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot average validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(epochs, avg_values, 'b-', marker='o', label='Avg Validation Accuracy')
    if best_avg_epoch in epochs:
        best_value = avg_values[epochs.index(best_avg_epoch)]
        plt.axvline(x=best_avg_epoch, color='r', linestyle='--', 
                   label=f'Best Epoch: {best_avg_epoch+1} (Acc: {best_value:.2f}%)')
        plt.plot(best_avg_epoch, best_value, 'ro', markersize=8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Average Validation Accuracy Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot standard deviation
    if std_values:
        plt.subplot(2, 1, 2)
        plt.plot(epochs, std_values, 'g-', marker='o', label='Std Dev of Validation Accuracy')
        if best_avg_epoch in epochs:
            best_std = std_values[epochs.index(best_avg_epoch)]
            plt.axvline(x=best_avg_epoch, color='r', linestyle='--', 
                       label=f'Best Epoch: {best_avg_epoch+1} (Std: {best_std:.2f}%)')
            plt.plot(best_avg_epoch, best_std, 'ro', markersize=8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Standard Deviation (%)')
        plt.title('Standard Deviation of Validation Accuracy Across Folds')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Display best epoch metrics
    if best_avg_epoch in epochs:
        idx = epochs.index(best_avg_epoch)
        best_avg = avg_values[idx]
        best_std = std_values[idx] if std_values else 0
        
        print(f"\nBest epoch metrics (Epoch {best_avg_epoch+1}):")
        print(f"  Average validation accuracy: {best_avg:.2f}%")
        if std_values:
            print(f"  Standard deviation: {best_std:.2f}%")
            # Calculate confidence intervals
            k = int(run.data.params.get("k_folds", 3))
            confidence_level = 0.8  # 80% confidence level used in your study
            t_critical = stats.t.ppf(confidence_level, df=k-1)
            lower_bound = best_avg - (t_critical * best_std / np.sqrt(k))
            upper_bound = best_avg + (t_critical * best_std / np.sqrt(k))
            print(f"  {confidence_level*100:.0f}% confidence interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
            print(f"  Lower bound (study objective): {lower_bound:.2f}%")

