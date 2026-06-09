# Transformer-Level Load Forecasting — João Pessoa, Brazil

Code for the paper: *"Electricity Demand Prediction at Power Transformers in Urban Distribution Systems: A Comparative Medium-Term Forecasting Study"*

## What this is

365-day-ahead forecasting of daily peak apparent power (Smax) for three distribution transformers in João Pessoa, Brazil (2021–2024). Eight models are compared under univariate (UV) and multivariate (MV) configurations:

- **SARIMAX** — Statistical baseline
- **SVR** — Kernel-based
- **LightGBM, XGBoost, Gradient Boosting** — Gradient boosting ensemble
- **LSTM, CNN-LSTM** — Deep learning (recurrent / hybrid)
- **N-BEATS** — Neural basis expansion (deep learning)

## Files

| File | Description |
|------|-------------|
| `config.py` | All paths, dates, and hyperparameters |
| `utils.py` | Shared functions: loading, metrics, plots, bootstrap CI, Diebold-Mariano test |
| `EDA.py` | Exploratory analysis |
| `models_sarimax.py` | SARIMAX |
| `models_svr.py` | SVR (UV + MV) |
| `models_boosting.py` | LightGBM, XGBoost, GradientBoosting (UV + MV) |
| `models_dl.py` | LSTM, CNN-LSTM (UV + MV) |
| `models_nbeats.py` | N-BEATS (UV + MV) |
| `requirements.txt` | Python dependencies |

## Data

Place files in a `data/` folder next to the scripts:

- **T1.csv** — commercial transformer
- **T2.csv** — residential transformer
- **T3.csv** — coastal mixed-use transformer
- **weather.csv** — climatological averages (optional, needed for MV)

Each transformer CSV has: `date` (DD/MM/YYYY) and `Smax` (normalised to [0, 1]).

Weather CSV uses `;` as separator with columns:
`date`, `temp_max_anos`, `temp_min_anos`, `temp_media_anos`, `precipitacao_anos`

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.10 and 3.11.

## Running

```bash
python EDA.py               # exploratory analysis
python models_sarimax.py    # SARIMAX baseline
python models_svr.py        # SVR
python models_boosting.py   # gradient boosting (LightGBM, XGBoost, GB)
python models_dl.py         # LSTM + CNN-LSTM
python models_nbeats.py     # N-BEATS
```

Results are saved in `results/` subfolders created automatically. If `data/weather.csv` is absent, all models run in UV mode without error.

## Experimental Design

| Split | Period | Days |
|-------|--------|------|
| Training | 2021-01-01 to 2022-12-31 | ~730 |
| Validation | 2023-01-01 to 2023-12-31 | 365 |
| Test | 2024-01-01 to 2024-12-31 | 365 |

- Feature selection (MV) uses training data only, before HPT
- Validation used solely for hyperparameter tuning
- Test set accessed exactly once per model, after HPT
- Scalers fitted on training set only
- Deep learning models trained N_DL_RUNS=5 times with distinct seeds; metrics reported as mean ± std

## Key Results

| Transformer | Profile | Best model | sMAPE | R² |
|-------------|---------|-----------|-------|----|
| T1 | Commercial | CNN-LSTM UV | 10.46% | 0.704 |
| T2 | Residential | CNN-LSTM UV | 4.05% | 0.715 |
| T3 | Coastal mixed | LSTM UV | 5.87% | 0.532 |

MV improves gradient boosting by 5–30% RMSE. Effect on deep learning is architecture- and transformer-specific.
