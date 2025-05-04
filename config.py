import torch
import numpy as np
import random

# Default configuration values
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths (will be populated in notebook)
TRAIN_PATHS = []
TEST_PATHS = []

# Class information (will be populated in notebook)
NUM_CLASSES = 0
CLASS_NAMES = []
CLASS_WEIGHTS = None

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")