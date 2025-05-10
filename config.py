"""
Configuration for CK+ Subject training
"""

# Data settings
CK_PATH = "data/aligned_ck_data.pkl"
KDEF_PATH = "data/aligned_kdef_data.pkl"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

# Model settings
IN_CHANNELS = 1
MODEL_CHANNELS = 8
OUT_CHANNELS = 1
CONTEXT_DIM = 8
NUM_RES_BLOCKS = 1
ATTENTION_RESOLUTIONS = "20"
DROPOUT = 0.15
USE_SCALE_SHIFT_NORM = True

# Specify aggressive downsampling with custom channel multipliers
# 8->32->64->256->? progression
CHANNEL_MULT = (1, 2, 8, 16, 32, 64)

# Flow matching settings
SIGMA = 0.1
FLOW_MATCHER_TYPE = ''

# Perceptual loss settings
USE_PERCEPTUAL_LOSS = True
PERCEPTUAL_LOSS_WEIGHT = 0.01  # Reduced from 0.01 to better balance with other losses
PERCEPTUAL_LOSS_DELAY_EPOCHS = 1 # Reduced from 100 to apply perceptual loss earlier
EULER_STEPS = 5

# LPIPS loss settings
USE_LPIPS_LOSS = True
LPIPS_LOSS_WEIGHT = 0.05  # Reduced from 0.1 to better balance with other losses
LPIPS_LOSS_DELAY_EPOCHS = 1
LPIPS_NET = 'vgg'  # Network backbone for LPIPS ('vgg', 'alex', or 'squeeze')

# Training settings
SEED = 42
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 100
GRAD_CLIP_NORM = 1.5
EVALUATION_INTERVAL = 3
SAVE_BEST_ONLY = True

# Learning rate scheduler settings
USE_LR_SCHEDULER = False
WARMUP_EPOCHS = 10  # Linear warmup period
MIN_LR = 1e-7  # Minimum learning rate for cosine annealing

# Checkpoint settings
CHECKPOINT_DIR = 'checkpoints/'
SAVE_INTERVAL = 1

# Hardware settings
USE_CUDA = True

# Logging settings
LOG_TO_WANDB = True
WANDB_PROJECT = 'emoflow'
EXPERIMENT_NAME = 'experiment'
LOG_INTERVAL = 10

