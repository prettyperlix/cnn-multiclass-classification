"""
Configuration parameters for the Deep Learning project.
"""
import os

# Dataset paths - UPDATE THIS TO YOUR DATASET FOLDER
BASE_DIR = "fish_image" 

# Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Keep it at 16 or 32 to avoid Out of Memory (OOM) errors
LEARNING_RATE = 0.001
EPOCHS = 50      # High number, but Early Stopping will halt it when optimal
DROPOUT_RATE = 0.5

# Model saving paths
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True) # Creates the folder if it doesn't exist

CUSTOM_CNN_PATH = os.path.join(MODELS_DIR, "custom_cnn.keras")
MOBILENET_PATH = os.path.join(MODELS_DIR, "mobilenet_scratch.keras")
RESNET_PATH = os.path.join(MODELS_DIR, "resnet_transfer.keras")
INCEPTION_PATH = os.path.join(MODELS_DIR, "inception_transfer.keras")