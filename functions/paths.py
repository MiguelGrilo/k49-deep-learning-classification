from pathlib import Path

# .parent = 'functions'
# .parent.parent = root (FinalProject_58387_58656)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / 'dataset'
YOLO_FORMAT_DIR = DATASET_DIR / 'yolo_format'
KERAS_TUNER_DIR = ROOT_DIR / 'keras_tuner'
RESULTS_DIR = ROOT_DIR / 'results'
WEIGHTS_DIR = ROOT_DIR / 'weights'