import os

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ARTIF_DIR = "artifacts"
MODELS_DIR = os.path.join(ARTIF_DIR, "models")
RUNS_DIR = os.path.join(ARTIF_DIR, "runs")

SEED = 42
MAX_SEQ_LEN = 1024  # max token length