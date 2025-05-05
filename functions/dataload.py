
from config import SEED, CLASS_NAMES
import os
import random
import collections
import webdataset as wds
from torchvision import transforms


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
