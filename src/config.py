"""Configuration settings."""

from pathlib import Path
import torch

# --- Core Configuration ---

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
SEED_START_VALUE = 0
NUM_INFERENCE_STEPS = 30

# --- Directory Setup ---
OUTPUT_DIR = Path("output")
IMAGE_DIR = OUTPUT_DIR / "images"
ALL_IMAGE_DIR = OUTPUT_DIR / "all_images"
ENTITIES_DIR = Path("entities")

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ALL_IMAGES = True  # Whether to save all generated images and heatmaps
OUTPUT_IMAGE_DIMENSIONS = (256, 256)  # Default image dimensions for saving
if OUTPUT_ALL_IMAGES:
    ALL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- Entity Loading ---
with open(ENTITIES_DIR / "objects.txt", "r", encoding="utf-8") as f:
    CONTENTS = f.read().splitlines()

with open(ENTITIES_DIR / "movements.txt", "r", encoding="utf-8") as f:
    MOVEMENTS = [style.split()[1].replace("_", " ") for style in f.read().splitlines()]

with open(ENTITIES_DIR / "artists.txt", "r", encoding="utf-8") as f:
    ARTISTS = [artist.split()[1].replace("_", " ") for artist in f.read().splitlines()]

STYLES = MOVEMENTS + ARTISTS
MEDIUMS = ["painting"]

# --- Prompt & Experiment Parameters ---
PROMPT_TEMPLATES = [
    "a <MEDIUM> of a <CONTENT> in the <STYLE> style",
    "a <STYLE> <MEDIUM> of a <CONTENT>",
    "a <CONTENT> in the <STYLE> style",
    "a <CONTENT> with <STYLE> style",
]

IOU_THRESHOLDS_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# --- Post-processing Configuration for Image Saving ---
N_RELEVANT_IMAGES_TO_SAVE = 10  # Number of top/bottom images
DEFAULT_IOU_THRESHOLD_FOR_VIS = [0.4, 0.5, 0.8, 0.9]
DEFAULT_USE_QUANTILE_FOR_VIS = [False, False, True, True]
