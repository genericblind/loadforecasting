# =============================================================================
# config.py — Central configuration
# Transformer-level load forecasting — João Pessoa, Brazil
# =============================================================================
# Edit only this file to change transformer IDs, dates, or hyperparameters.
# All scripts import from here, ensuring full consistency.
# =============================================================================

import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# Place CSV files in a data/ folder next to the scripts
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = "data"
WEATHER_PATH = os.path.join(DATA_DIR, "weather.csv")

# Each transformer has its own CSV: T1.csv, T2.csv, T3.csv
# Columns: date (DD/MM/YYYY), Smax (normalised to [0, 1])
TRANSFORMER_FILES = {
    "T1": os.path.join(DATA_DIR, "T1.csv"),
    "T2": os.path.join(DATA_DIR, "T2.csv"),
    "T3": os.path.join(DATA_DIR, "T3.csv"),
}

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_EDA         = "results/eda"
OUTPUT_UV          = "results/boosting_uv"
OUTPUT_MV          = "results/boosting_mv"
OUTPUT_DL          = "results/dl"
OUTPUT_SARIMAX     = "results/sarimax"
OUTPUT_SVR         = "results/svr"
OUTPUT_FUTURE      = "results/future"

for d in [OUTPUT_EDA, OUTPUT_UV, OUTPUT_MV, OUTPUT_DL,
          OUTPUT_SARIMAX, OUTPUT_SVR, OUTPUT_FUTURE]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMERS
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMADORES = ["T1", "T2", "T3"]

MAPA_PLOT = {
    "T1": "T1",
    "T2": "T2",
    "T3": "T3",
}

# ─────────────────────────────────────────────────────────────────────────────
# TIME SPLITS — no data leakage
#
#   ┌─────────────────┬──────────────┬─────────────┐
#   │    TRAIN        │  VALIDATION  │    TEST     │
#   │  2021 – 2022    │    2023      │    2024     │
#   └─────────────────┴──────────────┴─────────────┘
# ─────────────────────────────────────────────────────────────────────────────
START_DATE     = "2021-01-01"
END_TRAIN      = "2022-12-31"
END_VALIDATION = "2023-12-31"

# ─────────────────────────────────────────────────────────────────────────────
# FUTURE FORECAST
# ─────────────────────────────────────────────────────────────────────────────
FUTURE_END_DATE = "2027-12-31"

# ─────────────────────────────────────────────────────────────────────────────
# TIME SERIES PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
PERIOD         = 365
MAX_GAP_INTERP = 7
LAGS           = [1, 2, 7, 14, 30, 60, 180, 365]
ROLLING_WINDOW = 365
STEPS          = 365

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING (LSTM / CNN-LSTM)
# ─────────────────────────────────────────────────────────────────────────────
DL_LOOKBACK    = 365
DL_HORIZON     = 1
DL_EPOCHS      = 150
DL_BATCH_SIZE  = 32
DL_PATIENCE    = 20
DL_LR          = 1e-3

N_DL_RUNS = 5
DL_SEEDS  = [42, 123, 456, 789, 2024]
assert len(DL_SEEDS) == N_DL_RUNS

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK BOOTSTRAP — confidence intervals
# ─────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_BLOCK_SIZE  = 30
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_ALPHA       = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# SVR
# ─────────────────────────────────────────────────────────────────────────────
SVR_LAGS = [1, 2, 7, 14, 30, 60, 180, 365]

# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX — HPT order grid
# ─────────────────────────────────────────────────────────────────────────────
SARIMAX_ORDERS = [
    ((1,1,1), (1,1,1,7)),
    ((2,1,1), (1,1,1,7)),
    ((1,1,2), (1,1,1,7)),
    ((2,1,2), (0,1,1,7)),
]

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
