"""
Configuration for the facial expression editing Gradio app
"""

# Path settings
CHECKPOINT_DIR = "checkpoints"

# Dataset settings
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2

# Model architecture
IN_CHANNELS = 1  # Grayscale images
MODEL_CHANNELS = 8  # Base dimension for the model
NUM_RES_BLOCKS = 1
ATTENTION_RESOLUTIONS = "20"
DROPOUT = 0.15
CHANNEL_MULT = (1, 2, 8, 16, 32, 64)  # Channel multipliers for each resolution level
CONTEXT_DIM = 8  # Number of emotion classes
USE_SCALE_SHIFT_NORM = True

# For trajectory visualization
EULER_STEPS = 10  # Number of steps in the trajectory

# Device settings
USE_CUDA = True
