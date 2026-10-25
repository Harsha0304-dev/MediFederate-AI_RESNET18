"""Backward-compatible project configuration."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/'data'; RESULTS_DIR=ROOT/'results'; MODELS_DIR=ROOT/'models'
XRAY_TRAIN=DATA_DIR/'processed'/'xray'/'train'
XRAY_VAL=DATA_DIR/'processed'/'xray'/'validation'
XRAY_TEST=DATA_DIR/'processed'/'xray'/'test'
IMAGE_SIZE=224
NUM_CLASSES=2
CLASS_NAMES=['NORMAL','PNEUMONIA']
SEED=42
