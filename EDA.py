# =============================================================================
# EDA.py — Exploratory Data Analysis
# =============================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import (
    TRANSFORMADORES, MAPA_PLOT, START_DATE, PERIOD, OUTPUT_EDA
)
from utils import (
    carregar_dados, carregar_weather, preparar_serie_diaria,
    adicionar_features_calendario, log
)

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.titlesize": 14,
    "axes.titlepad":    8,
    "figure.dpi":       160,
    "savefig.dpi":      160,
    "savefig.bbox":     "tight",
})


def savefig(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_series_overview(y: pd.Series, tid: str):
    """Raw time series with rolling statistics."""
    nome = MAPA_PLOT.get(tid, tid)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y.index, y.values, lw=0.8, color="#2563eb", label="Daily Smax")
    ax.plot(y.rolling(30).mean(), lw=1.5, color="#dc2626", label="30-day MA")
    ax.set_title(f"{nome} — Daily peak apparent power (Smax)", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("$S_{\\max}$ (normalized)", fontsize=12)
    ax.legend(fontsize=11)
    savefig(f"{OUTPUT_EDA}/series_{tid}.svg")
    log.info(f"  {tid}: series plot saved")


def plot_stl_decomposition(y: pd.Series, tid: str):
    """STL decomposition (trend + seasonality + residual)."""
    nome   = MAPA_PLOT.get(tid, tid)
    result = STL(y, period=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, data, label, color in zip(
        axes,
        [y, result.trend, result.seasonal, result.resid],
        ["Observed", "Trend", "Seasonal (weekly)", "Residual"],
        ["#2563eb", "#16a34a", "#f59e0b", "#6366f1"]
    ):
        ax.plot(data.index, data.values, lw=0.8, color=color)
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
    axes[0].set_title(f"{nome} — STL Decomposition", fontsize=13)
    axes[-1].set_xlabel("Date", fontsize=12)
    plt.tight_layout()
    savefig(f"{OUTPUT_EDA}/stl_{tid}.svg")
    log.info(f"  {tid}: STL plot saved")


def plot_autocorrelation(y: pd.Series, tid: str, lags: int = 60):
    """ACF and PACF plots."""
    nome = MAPA_PLOT.get(tid, tid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(y.dropna(),  ax=axes[0], lags=lags, alpha=0.05)
    plot_pacf(y.dropna(), ax=axes[1], lags=lags, alpha=0.05)
    axes[0].set_title(f"{nome} — ACF",  fontsize=13)
    axes[1].set_title(f"{nome} — PACF", fontsize=13)
    for ax in axes:
        ax.set_xlabel("Lag (days)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    savefig(f"{OUTPUT_EDA}/acf_{tid}.svg")
    log.info(f"  {tid}: ACF/PACF plot saved")


def plot_seasonal_boxplots(y: pd.Series, tid: str):
    """Boxplots by weekday and month."""
    nome = MAPA_PLOT.get(tid, tid)
    df   = pd.DataFrame({"Smax": y, "weekday": y.index.weekday, "month": y.index.month})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df.boxplot(column="Smax", by="weekday", ax=axes[0],
               boxprops=dict(color="#2563eb"),
               medianprops=dict(color="#dc2626", lw=2))
    axes[0].set_title(f"{nome} — Smax by Weekday", fontsize=13)
    axes[0].set_xlabel("Weekday (0=Mon, 6=Sun)", fontsize=12)
    axes[0].set_ylabel("$S_{\\max}$ (normalized)", fontsize=12)

    df.boxplot(column="Smax", by="month", ax=axes[1],
               boxprops=dict(color="#2563eb"),
               medianprops=dict(color="#dc2626", lw=2))
    axes[1].set_title(f"{nome} — Smax by Month", fontsize=13)
    axes[1].set_xlabel("Month", fontsize=12)
    axes[1].set_ylabel("$S_{\\max}$ (normalized)", fontsize=12)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=10)

    plt.suptitle("")
    plt.tight_layout()
    savefig(f"{OUTPUT_EDA}/seasonal_boxplots_{tid}.svg")
    log.info(f"  {tid}: seasonal boxplots saved")


def plot_weather_correlation(y: pd.Series, weather: pd.DataFrame, tid: str):
    """Scatter plots of Smax vs weather variables."""
    nome = MAPA_PLOT.get(tid, tid)
    df   = pd.DataFrame({"Smax": y}).join(weather, how="inner").dropna()

    wcols = [c for c in weather.columns if c in df.columns]
    if not wcols:
        return

    n    = len(wcols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, wcols):
        ax.scatter(df[col], df["Smax"], alpha=0.3, s=10, color="#2563eb")
        corr = df[[col, "Smax"]].corr().iloc[0, 1]
        ax.set_title(f"{nome}: {col}\n(r = {corr:.3f})", fontsize=12)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("$S_{\\max}$ (normalized)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    savefig(f"{OUTPUT_EDA}/weather_corr_{tid}.svg")
    log.info(f"  {tid}: weather correlation plot saved")


def summary_statistics(y: pd.Series, tid: str) -> dict:
    """Compute and print basic descriptive statistics."""
    nome = MAPA_PLOT.get(tid, tid)
    stats = {
        "transformer": nome,
        "n_days":      len(y),
        "start":       str(y.index.min().date()),
        "end":         str(y.index.max().date()),
        "mean":        round(float(y.mean()), 4),
        "std":         round(float(y.std()), 4),
        "min":         round(float(y.min()), 4),
        "max":         round(float(y.max()), 4),
        "missing_pct": round(float(y.isna().mean()) * 100, 2),
    }
    log.info(f"  {tid}: {stats}")
    return stats


def main():
    df = carregar_dados()
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    # Optional weather data
    weather = None
    try:
        weather = carregar_weather()
        log.info("Weather data loaded.")
    except Exception as e:
        log.warning(f"Weather not loaded (MV EDA skipped): {e}")

    all_stats = []
    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*50}\n  EDA — {tid}\n{'='*50}")
        y = preparar_serie_diaria(df_tr)

        stats = summary_statistics(y, tid)
        all_stats.append(stats)

        plot_series_overview(y, tid)
        plot_stl_decomposition(y, tid)
        plot_autocorrelation(y, tid)
        plot_seasonal_boxplots(y, tid)

        if weather is not None:
            plot_weather_correlation(y, weather, tid)

    pd.DataFrame(all_stats).to_csv(f"{OUTPUT_EDA}/summary_statistics.csv", index=False)
    log.info(f"\nEDA complete. Outputs saved to '{OUTPUT_EDA}/'")


if __name__ == "__main__":
    main()
