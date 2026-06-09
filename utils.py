# =============================================================================
# utils.py — Shared utility functions
# =============================================================================
import warnings, logging, os
import numpy as np
import pandas as pd
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import (
    START_DATE, END_TRAIN, END_VALIDATION,
    MAX_GAP_INTERP, RANDOM_STATE, PERIOD, MAPA_PLOT,
    TRANSFORMER_FILES, WEATHER_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size":             12,
    "axes.titlesize":        13,
    "axes.labelsize":        12,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "legend.fontsize":       11,
    "legend.title_fontsize": 11,
    "figure.titlesize":      14,
    "axes.titlepad":         8,
    "figure.dpi":            150,
    "savefig.dpi":           150,
    "savefig.bbox":          "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def carregar_dados() -> pd.DataFrame:
    """
    Load all transformer CSVs and return a single DataFrame with an 'id' column.
    Each CSV must have columns: date (DD/MM/YYYY) and Smax (normalised to [0,1]).
    """
    frames = []
    for tid, path in TRANSFORMER_FILES.items():
        if not os.path.exists(path):
            log.warning(f"File not found: {path} — skipping {tid}")
            continue
        df = pd.read_csv(path)
        df["id"] = tid
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No transformer data files found in {list(TRANSFORMER_FILES.values())}")
    return pd.concat(frames, ignore_index=True)


def carregar_weather() -> pd.DataFrame:
    """
    Load weather CSV. Separator: semicolon (;).
    Expected columns: date, temp_max_anos, temp_min_anos, temp_media_anos, precipitacao_anos
    Returns a DataFrame indexed by date with English column names.
    """
    if not os.path.exists(WEATHER_PATH):
        raise FileNotFoundError(f"Weather file not found: {WEATHER_PATH}")
    w = pd.read_csv(WEATHER_PATH, sep=";")
    w["date"] = pd.to_datetime(w["date"], dayfirst=True)
    w = w.set_index("date").sort_index()
    rename_map = {
        "temp_max_anos":     "temp_max_hist",
        "temp_min_anos":     "temp_min_hist",
        "temp_media_anos":   "temp_mean_hist",
        "precipitacao_anos": "precip_hist",
    }
    cols_in = [c for c in rename_map if c in w.columns]
    w = w[cols_in].rename(columns=rename_map)
    return w.copy()


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE, sMAPE (symmetric, recommended for papers), R²."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, sMAPE=np.nan, R2=np.nan)
    eps   = 1e-8
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    mape  = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    ) * 100)
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, sMAPE=smape, R2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preparar_serie_diaria(df_tr: pd.DataFrame, log_gaps: bool = True) -> pd.Series:
    """
    Build a continuous daily series with interpolation limited to MAX_GAP_INTERP days.
    Accepts both 'datahora' (legacy) and 'date' column names for the timestamp.
    """
    s = df_tr.copy()
    time_col = "date" if "date" in s.columns else "datahora"
    s["Time"] = pd.to_datetime(s[time_col], dayfirst=True)
    s = s.set_index("Time").sort_index()
    full_idx = pd.date_range(s.index.min().normalize(),
                             s.index.max().normalize(), freq="D")
    s = s.reindex(full_idx).loc[pd.to_datetime(START_DATE):]

    if log_gaps:
        missing = s["Smax"].isna()
        if missing.any():
            grp  = (missing != missing.shift()).cumsum()
            lens = missing[missing].groupby(grp[missing]).count()
            large = lens[lens > MAX_GAP_INTERP]
            if not large.empty:
                log.warning(f"Gaps > {MAX_GAP_INTERP} days: {large.to_dict()}")

    s["Smax"] = s["Smax"].interpolate(method="time", limit=MAX_GAP_INTERP).ffill(limit=2)
    return s["Smax"]


def adicionar_features_calendario(serie: pd.Series) -> pd.DataFrame:
    """Calendar features with sin/cos cyclic encoding."""
    idx = serie.index
    br  = holidays.Brazil()
    df  = pd.DataFrame(index=idx)
    df["weekday"]       = idx.weekday
    df["month"]         = idx.month
    df["dayofyear"]     = idx.dayofyear
    df["year"]          = idx.year
    df["weekday_sin"]   = np.sin(2*np.pi*df["weekday"]   / 7)
    df["weekday_cos"]   = np.cos(2*np.pi*df["weekday"]   / 7)
    df["month_sin"]     = np.sin(2*np.pi*df["month"]     / 12)
    df["month_cos"]     = np.cos(2*np.pi*df["month"]     / 12)
    df["dayofyear_sin"] = np.sin(2*np.pi*df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2*np.pi*df["dayofyear"] / 365)
    df["is_weekend"]    = (idx.weekday >= 5).astype(int)
    df["is_holiday"]    = idx.isin(br).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SPLITS — no data leakage
# ─────────────────────────────────────────────────────────────────────────────
def get_splits(y: pd.Series) -> dict:
    """Train / Validation / Test — non-overlapping temporal windows."""
    t = pd.to_datetime(END_TRAIN)
    v = pd.to_datetime(END_VALIDATION)
    tr  = y.index[y.index <= t]
    val = y.index[(y.index > t) & (y.index <= v)]
    te  = y.index[y.index > v]
    log.info(f"Split → Train:{len(tr)} | Val:{len(val)} | Test:{len(te)} days")
    return dict(train=tr, val=val, test=te)


def build_lag_matrix(y: pd.Series, lags: list) -> pd.DataFrame:
    """Build lag feature matrix for tabular models (SVR, XGBoost)."""
    df = pd.DataFrame({"y": y})
    for lag in lags:
        df[f"lag_{lag}"] = y.shift(lag)
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_forecast_3way(y, pred_val, pred_test, tid, modelo, output_dir, suffix=""):
    """Standard paper figure: train / validation / test + forecasts."""
    nome  = MAPA_PLOT.get(tid, tid)
    t_end = pd.to_datetime(END_TRAIN)
    v_end = pd.to_datetime(END_VALIDATION)

    fig, ax = plt.subplots(figsize=(14, 5))
    y.loc[:t_end].plot(ax=ax, label="Training",     color="#2563eb", lw=1.2)
    y.loc[t_end:v_end].plot(ax=ax, label="Validation", color="#f59e0b", lw=1.2)
    y.loc[v_end:].plot(ax=ax, label="Test (real)",  color="#16a34a", lw=1.5)
    if pred_val  is not None and len(pred_val)  > 0:
        pred_val.plot(ax=ax,  label="Val. forecast", color="#f59e0b", lw=1.5, ls="--")
    if pred_test is not None and len(pred_test) > 0:
        pred_test.plot(ax=ax, label="Test forecast", color="#dc2626", lw=1.5, ls="--")
    ax.axvline(t_end, color="gray", ls=":", lw=1)
    ax.axvline(v_end, color="gray", ls=":", lw=1)
    ax.set_title(f"{nome} — {modelo}{' '+suffix if suffix else ''}", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("$S_{\\max}$ (normalized)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    path = f"{output_dir}/forecast_{tid}_{modelo}{suffix}.svg"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_residuals(y_true, y_pred, tid, modelo, output_dir):
    """Residuals over time + histogram."""
    nome = MAPA_PLOT.get(tid, tid)
    res  = np.asarray(y_true) - np.asarray(y_pred)
    idx  = getattr(y_true, "index", range(len(res)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(idx, res, lw=0.9, color="#6366f1")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_title(f"{nome} — {modelo}: Residuals over Time", fontsize=13)
    axes[0].set_xlabel("Date", fontsize=12)
    axes[0].set_ylabel("Residual ($S_{\\max}$ normalized)", fontsize=12)
    axes[0].tick_params(axis="both", labelsize=11)

    axes[1].hist(res, bins=40, color="#6366f1", edgecolor="white")
    axes[1].set_title(f"{nome} — {modelo}: Residual Distribution", fontsize=13)
    axes[1].set_xlabel("Residual ($S_{\\max}$ normalized)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_{tid}_{modelo}.svg",
                dpi=150, bbox_inches="tight"); plt.close()


def plot_future_forecast(y_hist, future_mean, future_lower, future_upper,
                         tid, modelo, output_dir):
    """Future forecast with 95% bootstrap confidence interval."""
    nome = MAPA_PLOT.get(tid, tid)
    fig, ax = plt.subplots(figsize=(16, 5))
    y_hist.plot(ax=ax, label="Historical", color="#2563eb", lw=1.2)
    ax.plot(future_mean.index, future_mean.values,
            label="Future forecast", color="#dc2626", lw=2, ls="--")
    ax.fill_between(future_mean.index, future_lower, future_upper,
                    alpha=0.2, color="#dc2626", label="95% CI (bootstrap)")
    ax.axvline(y_hist.index[-1], color="gray", ls=":", lw=1.5, label="Last observed")
    ax.set_title(f"{nome} — {modelo}: Future Forecast with 95% CI", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("$S_{\\max}$ (normalized)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/future_{tid}_{modelo}.svg",
                dpi=150, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK BOOTSTRAP — confidence intervals for test metrics
# ─────────────────────────────────────────────────────────────────────────────
def block_bootstrap_metric_ci(y_true, y_pred,
                               metric_fn,
                               block_size=30,
                               n_resamples=1000,
                               alpha=0.05,
                               random_state=42) -> dict:
    """
    Block bootstrap CI for an arbitrary metric.
    Resamples contiguous blocks of size `block_size` with replacement,
    preserving local autocorrelation structure of residuals.

    Reference: Künsch (1989). Annals of Statistics 17(3): 1217–1241.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n      = len(y_true)
    if n == 0:
        return dict(point=np.nan, lower=np.nan, upper=np.nan,
                    std=np.nan, samples=np.array([]))

    rng      = np.random.default_rng(random_state)
    point    = float(metric_fn(y_true, y_pred))
    n_blocks = int(np.ceil(n / block_size))
    samples  = np.empty(n_resamples, dtype=float)

    for r in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        try:
            samples[r] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            samples[r] = np.nan

    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return dict(point=point, lower=np.nan, upper=np.nan,
                    std=np.nan, samples=np.array([]))

    lower = float(np.percentile(samples, 100 * (alpha / 2)))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    std   = float(np.std(samples, ddof=1))
    return dict(point=point, lower=lower, upper=upper, std=std, samples=samples)


# ─────────────────────────────────────────────────────────────────────────────
# DIEBOLD-MARIANO TEST
# ─────────────────────────────────────────────────────────────────────────────
def diebold_mariano_test(y_true, pred1, pred2, h: int = 1, loss: str = "se") -> dict:
    """
    Diebold-Mariano test with HLN correction for finite samples.

    H0: both models have equal predictive accuracy.
    DM > 0 → model 2 is better; DM < 0 → model 1 is better.

    References:
      Diebold & Mariano (1995). J. Business & Economic Statistics 13(3): 253–263.
      Harvey, Leybourne & Newbold (1997). Int. J. Forecasting 13(2): 281–291.
    """
    from scipy.stats import t as t_dist

    y_true = np.asarray(y_true, dtype=float)
    pred1  = np.asarray(pred1,  dtype=float)
    pred2  = np.asarray(pred2,  dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(pred1) | np.isnan(pred2))
    y_true, pred1, pred2 = y_true[mask], pred1[mask], pred2[mask]
    n = len(y_true)
    if n < 10:
        return dict(DM=np.nan, p_value=np.nan, mean_d=np.nan, n=n, loss=loss)

    e1 = y_true - pred1
    e2 = y_true - pred2
    d  = e1**2 - e2**2 if loss == "se" else np.abs(e1) - np.abs(e2)

    mean_d  = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=0))

    if h > 1:
        autocov = [float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
                   for k in range(1, h)]
        var_d = (gamma_0 + 2.0 * sum(autocov)) / n
    else:
        var_d = gamma_0 / n

    if var_d <= 0:
        return dict(DM=np.nan, p_value=np.nan, mean_d=mean_d, n=n, loss=loss)

    DM         = mean_d / np.sqrt(var_d)
    hln_factor = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    DM_hln     = DM * hln_factor
    p_value    = float(2.0 * (1.0 - t_dist.cdf(abs(DM_hln), df=n-1)))
    return dict(DM=float(DM_hln), p_value=p_value, mean_d=mean_d, n=n, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL NAIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def seasonal_naive_forecast(y_train_val: pd.Series,
                             test_index: pd.DatetimeIndex,
                             period: int = 7) -> pd.Series:
    """Seasonal naive: ŷ_t = y_{t-period}. Recursive for multi-step."""
    full = pd.concat([y_train_val, pd.Series(np.nan, index=test_index)])
    for ts in test_index:
        ref = ts - pd.Timedelta(days=period)
        if ref in full.index:
            full.loc[ts] = full.loc[ref]
        else:
            full.loc[ts] = full.loc[y_train_val.index[-1]]
    return full.loc[test_index]
