"""Utility functions for data processing and analysis."""


# Define a function to monitor memory usage
def print_gpu_memory():
    """
    Print the current GPU memory usage.
    This function checks if a GPU is available and prints the allocated and reserved memory.
    """
    import torch
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

    import os
    import collections
    import torch
    import webdataset as wds
    from tqdm.auto import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="webdataset")

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