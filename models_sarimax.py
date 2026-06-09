# =============================================================================
# models_sarimax.py — SARIMAX statistical baseline
# =============================================================================
# SARIMAX is the canonical statistical benchmark for seasonal time series.
# HPT: order selection (p,d,q)(P,D,Q,s) by AIC on the TRAINING set.
# Final evaluation: TEST set (2024) — accessed once.
#
# Reference:
#   Box, Jenkins, Reinsel & Ljung (2015). Time Series Analysis (5th ed.). Wiley.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (
    TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    SARIMAX_ORDERS, END_TRAIN, END_VALIDATION,
    OUTPUT_SARIMAX, OUTPUT_FUTURE
)
from utils import (
    carregar_dados, preparar_serie_diaria, adicionar_features_calendario,
    compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast,
    seasonal_naive_forecast, log
)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def treinar_sarimax(df: pd.DataFrame) -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [SARIMAX]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        # ── Seasonal naive baselines ──────────────────────────────────────────
        y_tv_naive = y.loc[splits["train"].union(splits["val"])]
        for period, label in [(7, "Naive-7d"), (365, "Naive-365d")]:
            naive_pred    = seasonal_naive_forecast(y_tv_naive, splits["test"], period=period)
            naive_metrics = compute_metrics(y.loc[splits["test"]], naive_pred)
            log.info(f"  {label} → RMSE={naive_metrics['RMSE']:.4f} R²={naive_metrics['R2']:.4f}")
            resultados.append({
                "transformer": tid, "label": MAPA_PLOT.get(tid, tid),
                "model": label, "mode": "Naive",
                "best_params": "N/A", "AIC_train": np.nan,
                **{k: round(v, 4) for k, v in naive_metrics.items()},
            })

        y_train = y.loc[splits["train"]]
        y_val   = y.loc[splits["val"]]
        y_test  = y.loc[splits["test"]]
        y_tv    = pd.concat([y_train, y_val])

        # Cyclic calendar regressors capture annual seasonality
        # (s=365 is computationally infeasible; s=7 captures weekly pattern)
        cal_full = adicionar_features_calendario(y)
        exog_cols = ["dayofyear_sin", "dayofyear_cos",
                     "month_sin", "month_cos",
                     "weekday_sin", "weekday_cos",
                     "is_weekend", "is_holiday"]
        exog_full  = cal_full[exog_cols]
        exog_train = exog_full.loc[splits["train"]]
        exog_val   = exog_full.loc[splits["val"]]
        exog_test  = exog_full.loc[splits["test"]]
        exog_tv    = exog_full.loc[y_tv.index]

        # HPT: select order by AIC on training set
        best_aic, best_order, best_sorder = np.inf, None, None
        for order, sorder in SARIMAX_ORDERS:
            try:
                model = SARIMAX(y_train, exog=exog_train,
                                order=order, seasonal_order=sorder,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=200)
                log.info(f"     {order}x{sorder} → AIC={result.aic:.2f}")
                if result.aic < best_aic:
                    best_aic    = result.aic
                    best_order  = order
                    best_sorder = sorder
            except Exception as e:
                log.warning(f"  SARIMAX {order}x{sorder} failed: {e}")

        if best_order is None:
            log.error(f"  {tid}: no SARIMAX order converged.")
            continue

        log.info(f"  ✓ HPT → {best_order}x{best_sorder} AIC={best_aic:.2f}")

        # Validation forecast (for plot)
        try:
            m_val    = SARIMAX(y_train, exog=exog_train,
                               order=best_order, seasonal_order=best_sorder,
                               enforce_stationarity=False, enforce_invertibility=False)
            r_val    = m_val.fit(disp=False, maxiter=200)
            pred_val = r_val.forecast(steps=len(y_val), exog=exog_val)
            pred_val.index = y_val.index
        except Exception as e:
            log.warning(f"  Validation forecast failed: {e}")
            pred_val = pd.Series(np.nan, index=y_val.index)

        # Retrain on TRAIN+VAL, evaluate TEST
        try:
            m_tv      = SARIMAX(y_tv, exog=exog_tv,
                                order=best_order, seasonal_order=best_sorder,
                                enforce_stationarity=False, enforce_invertibility=False)
            r_tv      = m_tv.fit(disp=False, maxiter=200)
            pred_test = r_tv.forecast(steps=len(y_test), exog=exog_test)
            pred_test.index = y_test.index
        except Exception as e:
            log.error(f"  Test forecast failed: {e}")
            pred_test = pd.Series(np.nan, index=y_test.index)

        metrics = compute_metrics(y_test, pred_test)
        log.info(f"  ✓ TEST → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        plot_forecast_3way(y, pred_val, pred_test, tid, "SARIMAX", OUTPUT_SARIMAX)
        plot_residuals(y_test, pred_test, tid, "SARIMAX", OUTPUT_SARIMAX)

        pd.DataFrame({
            "date":   y_test.index,
            "y_true": y_test.values,
            "y_pred": pred_test.values,
        }).to_csv(f"{OUTPUT_SARIMAX}/preds_test_{tid}_SARIMAX.csv", index=False)

        # Residual diagnostics
        try:
            fig = r_tv.plot_diagnostics(figsize=(13, 9))
            fig.suptitle(f"{MAPA_PLOT.get(tid,tid)} — SARIMAX Residual Diagnostics",
                         fontsize=14)
            for ax in fig.axes:
                ax.tick_params(axis="both", labelsize=11)
                ax.xaxis.label.set_size(12)
                ax.yaxis.label.set_size(12)
                ax.title.set_size(12)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_SARIMAX}/diagnostics_{tid}_SARIMAX.svg",
                        dpi=150, bbox_inches="tight"); plt.close()
        except Exception:
            pass

        # Future forecast
        try:
            from config import FUTURE_END_DATE
            future_idx  = pd.date_range(y.index[-1] + pd.Timedelta(days=1),
                                        pd.to_datetime(FUTURE_END_DATE), freq="D")
            exog_future = adicionar_features_calendario(
                pd.Series(np.nan, index=future_idx)
            )[exog_cols]

            m_full = SARIMAX(y, exog=exog_full,
                             order=best_order, seasonal_order=best_sorder,
                             enforce_stationarity=False, enforce_invertibility=False)
            r_full = m_full.fit(disp=False, maxiter=200)
            fc_obj = r_full.get_forecast(steps=len(future_idx), exog=exog_future)
            pred_fut  = fc_obj.predicted_mean
            pred_fut.index = future_idx
            ci        = fc_obj.conf_int(alpha=0.05)
            lower_fut = pd.Series(ci.iloc[:, 0].values, index=future_idx)
            upper_fut = pd.Series(ci.iloc[:, 1].values, index=future_idx)

            plot_future_forecast(y, pred_fut, lower_fut, upper_fut,
                                 tid, "SARIMAX", OUTPUT_FUTURE)
            pd.DataFrame({
                "date":        future_idx,
                "pred_mean":   pred_fut.values,
                "ci_lower_95": lower_fut.values,
                "ci_upper_95": upper_fut.values,
            }).to_csv(f"{OUTPUT_FUTURE}/future_{tid}_SARIMAX.csv", index=False)
        except Exception as e:
            log.warning(f"  Future forecast failed: {e}")

        resultados.append({
            "transformer": tid, "label": MAPA_PLOT.get(tid, tid),
            "model": "SARIMAX", "mode": "Statistical",
            "best_params": f"{best_order}x{best_sorder}",
            "AIC_train":   round(best_aic, 2),
            **{k: round(v, 4) for k, v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    df = carregar_dados()
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    res = treinar_sarimax(df)
    res.to_csv(f"{OUTPUT_SARIMAX}/metrics_SARIMAX.csv", index=False)
    print(res[["transformer", "model", "RMSE", "MAE", "sMAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
