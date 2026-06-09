# =============================================================================
# models_nbeats.py — N-BEATS (Neural Basis Expansion Analysis for Time Series)
# =============================================================================
# N-BEATS: Oreshkin et al. (2020) — "N-BEATS: Neural basis expansion analysis
# for interpretable time series forecasting." ICLR 2020.
#
# Estrutura espelhada ao models_dl.py para comparação simétrica.
#
# Implementação: N-BEATS genérico (generic basis) em Keras puro.
# Não usa bibliotecas externas como neuralforecast para manter consistência
# com o restante do projeto (TensorFlow 2.15 / Keras).
#
# Arquitetura:
#   - Múltiplos stacks, cada um com múltiplos blocos
#   - Cada bloco: FC layers → projeção de backcast e forecast
#   - Saída: soma dos forecasts de todos os blocos (residual stacking)
#   - Lookback: DL_LOOKBACK (365 dias) — idêntico ao LSTM/CNN-LSTM
#
# Modo UV:  entrada = janela (lookback, 1) — só Smax normalizado
# Modo MV:  entrada = janela (lookback, 1 + n_exog_sel) — Smax + features
#
# Previsão multi-step: recursiva (1 step por vez), idêntico ao DL.
# Scalers fitados APENAS no treino → sem data leakage.
# HPT avalia MAE na validação (escala original).
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
try:
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Subtract, Add, Flatten, Lambda
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    import keras.backend as K
except ImportError:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Subtract, Add, Flatten, Lambda
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    DL_LOOKBACK, DL_EPOCHS, DL_BATCH_SIZE, DL_PATIENCE, DL_LR,
    END_TRAIN, END_VALIDATION, OUTPUT_DL, OUTPUT_FUTURE, FUTURE_END_DATE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
# NOTE: All three random seeds are fixed globally to ensure full reproducibility.
# Results reported in the paper correspond to a single deterministic run.

EXOG_COLS = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos", "is_weekend", "is_holiday",
    "temp_max_hist", "temp_min_hist", "temp_mean_hist", "precip_hist"
]
FEATURE_IMPORTANCE_THRESHOLD = 0.95

# Output directory for N-BEATS results (reuses OUTPUT_DL folder)
OUTPUT_NBEATS = OUTPUT_DL


# ─────────────────────────────────────────────────────────────────────────────
# N-BEATS BLOCK
# ─────────────────────────────────────────────────────────────────────────────
def nbeats_block(x, input_size, theta_size, hidden_units, n_layers):
    """
    Single N-BEATS block.
    x: (batch, input_size)
    Returns: backcast (batch, input_size), forecast (batch, 1)
    """
    h = x
    for _ in range(n_layers):
        h = Dense(hidden_units, activation="relu")(h)

    # Generic basis: theta_b for backcast, theta_f for forecast
    theta = Dense(theta_size, activation="linear", use_bias=False)(h)

    # Split theta into backcast and forecast parts
    theta_b = Lambda(lambda t: t[:, :input_size])(theta)
    theta_f = Lambda(lambda t: t[:, input_size:])(theta)  # shape: (batch, 1)

    backcast = Dense(input_size, activation="linear", use_bias=False)(theta_b)
    forecast  = Dense(1,          activation="linear", use_bias=False)(theta_f)

    return backcast, forecast


# ─────────────────────────────────────────────────────────────────────────────
# N-BEATS MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_nbeats(lookback, n_feat=1, n_stacks=2, n_blocks_per_stack=3,
                 hidden_units=256, n_layers=4, theta_size=None):
    """
    N-BEATS generic architecture.

    Parameters
    ----------
    lookback : int
        Input sequence length (DL_LOOKBACK).
    n_feat : int
        Number of input channels (1 for UV, 1+n_exog for MV).
    n_stacks : int
        Number of stacks (each stack = n_blocks_per_stack blocks).
    n_blocks_per_stack : int
        Blocks per stack.
    hidden_units : int
        FC layer width within each block.
    n_layers : int
        Number of FC layers per block.
    theta_size : int or None
        Theta dimension; defaults to lookback + 1 (backcast + 1-step forecast).
    """
    if theta_size is None:
        theta_size = lookback + 1  # backcast_size + forecast_size (1 step)

    input_size = lookback * n_feat

    x_in = Input(shape=(lookback, n_feat), name="input")
    x_flat = Flatten()(x_in)          # (batch, lookback * n_feat)

    residual = x_flat
    forecasts = []

    for stack_id in range(n_stacks):
        for block_id in range(n_blocks_per_stack):
            backcast, forecast = nbeats_block(
                residual, input_size, theta_size, hidden_units, n_layers
            )
            residual = Subtract(name=f"residual_s{stack_id}_b{block_id}")(
                [residual, backcast]
            )
            forecasts.append(forecast)

    # Global forecast = sum of all block forecasts
    if len(forecasts) == 1:
        total_forecast = forecasts[0]
    else:
        total_forecast = Add(name="global_forecast")(forecasts)

    model = Model(inputs=x_in, outputs=total_forecast, name="NBEATS")
    model.compile(Adam(DL_LR), "mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HPT GRID — N-BEATS configs
# ─────────────────────────────────────────────────────────────────────────────
NBEATS_HPT_GRID = [
    # (n_stacks, n_blocks_per_stack, hidden_units, n_layers)
    {"n_stacks": 2, "n_blocks_per_stack": 3, "hidden_units": 128, "n_layers": 4},
    {"n_stacks": 2, "n_blocks_per_stack": 3, "hidden_units": 256, "n_layers": 4},
    {"n_stacks": 3, "n_blocks_per_stack": 3, "hidden_units": 256, "n_layers": 4},
]


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (idêntica ao models_dl.py)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features_mv(y_train, exog_train, tid, output_dir):
    proxy = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1,
        verbose=-1, importance_type="gain"
    )
    from skforecast.recursive import ForecasterRecursive
    from skforecast.preprocessing import RollingFeatures
    from config import LAGS, ROLLING_WINDOW
    fc = ForecasterRecursive(
        estimator=proxy, lags=LAGS,
        window_features=RollingFeatures(stats=["mean"], window_sizes=[ROLLING_WINDOW])
    )
    fc.fit(y=y_train, exog=exog_train)
    imp_df = fc.get_feature_importances()

    exog_imp = imp_df[imp_df["feature"].isin(exog_train.columns)].copy()
    exog_imp = exog_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    if exog_imp.empty or exog_imp["importance"].sum() == 0:
        return list(exog_train.columns)

    exog_imp["imp_cum"] = (exog_imp["importance"] / exog_imp["importance"].sum()).cumsum()
    selected = exog_imp.loc[
        exog_imp["imp_cum"].shift(1, fill_value=0) < FEATURE_IMPORTANCE_THRESHOLD,
        "feature"
    ].tolist()
    if not selected:
        selected = [exog_imp.iloc[0]["feature"]]

    log.info(f"  [FS-NBEATS] {tid}: {len(selected)}/{len(exog_imp)} → {selected}")

    # Feature importance plot
    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(exog_imp) * 0.5 + 1)))
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1], color=colors[::-1])
    ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — N-BEATS: Feature Selection\n"
                 f"Blue = selected ({len(selected)}/{len(exog_imp)})", fontsize=13)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_NBEATS_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# PREPARAÇÃO DE DADOS (idêntica ao models_dl.py)
# ─────────────────────────────────────────────────────────────────────────────
def preparar_dados_nbeats(df_tr, weather=None, mode="UV"):
    y = preparar_serie_diaria(df_tr)
    splits = get_splits(y)
    sel_features = []

    if mode == "MV":
        cal = adicionar_features_calendario(y)
        if weather is not None:
            exog_df = cal.join(weather, how="left").bfill().ffill()
        else:
            exog_df = cal
        cols_disp = [c for c in EXOG_COLS if c in exog_df.columns]
        exog_df = exog_df[cols_disp]

        tid = df_tr["id"].iloc[0] if "id" in df_tr.columns else "?"
        sel_features = selecionar_features_mv(
            y.loc[splits["train"]],
            exog_df.loc[splits["train"]],
            tid, OUTPUT_NBEATS
        )
        exog_sel = exog_df[[c for c in sel_features if c in exog_df.columns]]
        sc_exog = MinMaxScaler()
        sc_exog.fit(exog_sel.loc[splits["train"]].values)
        exog_sc = sc_exog.transform(exog_sel.values)
    else:
        exog_sc = None
        sc_exog = None

    # Target scaler — fit on train only
    y_raw = y.values.reshape(-1, 1).astype(np.float32)

    # Guard: fill any remaining NaN after interpolation with train median
    train_mask = y.index.isin(splits["train"])
    train_vals = y_raw[train_mask]
    if np.isnan(train_vals).any():
        median_train = float(np.nanmedian(train_vals))
        log.warning(f"  NaN in y_train — filling with median ({median_train:.4f})")
        y_raw = np.where(np.isnan(y_raw), median_train, y_raw)

    sc_y = MinMaxScaler()
    sc_y.fit(y_raw[train_mask])
    y_sc = sc_y.transform(y_raw).flatten().astype(np.float32)
    y_sc = np.nan_to_num(y_sc, nan=0.5, posinf=1.0, neginf=0.0)

    # Build feature matrix
    if exog_sc is not None:
        X_raw = np.hstack([y_sc.reshape(-1, 1), exog_sc]).astype(np.float32)
    else:
        X_raw = y_sc.reshape(-1, 1).astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.5, posinf=1.0, neginf=0.0)

    # Sliding windows
    def make_windows(X, y, lb):
        Xw, yw = [], []
        for i in range(lb, len(y)):
            Xw.append(X[i - lb:i])
            yw.append(y[i])
        return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)

    Xw, yw = make_windows(X_raw, y_sc, DL_LOOKBACK)
    idx_w  = y.index[DL_LOOKBACK:]

    t_end = pd.to_datetime(END_TRAIN)
    v_end = pd.to_datetime(END_VALIDATION)
    m_tr  = idx_w <= t_end
    m_vl  = (idx_w > t_end) & (idx_w <= v_end)
    m_te  = idx_w > v_end

    return dict(
        X_tr=Xw[m_tr], y_tr=yw[m_tr],
        X_vl=Xw[m_vl], y_vl=yw[m_vl],
        X_te=Xw[m_te], y_te=yw[m_te],
        idx_vl=idx_w[m_vl], idx_te=idx_w[m_te],
        sc_y=sc_y, sc_exog=sc_exog,
        y_series=y, n_feat=Xw.shape[2],
        X_all=Xw, y_all=yw, idx_all=idx_w,
        sel_features=sel_features,
        X_raw_full=X_raw,
        y_sc_full=y_sc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=DL_PATIENCE // 2, min_lr=1e-6, verbose=0),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE PREDICT (identical logic to models_dl.py)
# ─────────────────────────────────────────────────────────────────────────────
def recursive_predict_nbeats(model, last_window, fut_exog_sc, sc_y, steps):
    """
    UV: fut_exog_sc = None → slides window with previous prediction only.
    MV: fut_exog_sc = (steps, n_exog) → appends exogenous channel each step.
    """
    window   = last_window.copy()   # (lookback, n_feat)
    preds_sc = []
    n_feat   = window.shape[1]

    for s in range(steps):
        x_in = window[np.newaxis]                     # (1, lookback, n_feat)
        yhat = model.predict(x_in, verbose=0)[0, 0]
        preds_sc.append(yhat)

        new_row = np.zeros(n_feat, dtype=np.float32)
        new_row[0] = yhat
        if fut_exog_sc is not None and s < len(fut_exog_sc):
            new_row[1:] = fut_exog_sc[s]
        window = np.vstack([window[1:], new_row])

    preds_sc = np.array(preds_sc).reshape(-1, 1)
    return sc_y.inverse_transform(preds_sc).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def treinar_nbeats(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    """
    mode='UV' → input = Smax history only (n_feat=1)
    mode='MV' → input = Smax + selected exogenous features
    """
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [N-BEATS-{mode}]\n{'='*60}")
        data = preparar_dados_nbeats(df_tr, weather=weather, mode=mode)

        if len(data["X_tr"]) == 0 or len(data["X_vl"]) == 0:
            log.warning(f"  {tid}: insufficient data."); continue

        for key in ["X_tr", "y_tr", "X_vl", "y_vl", "X_te", "y_te", "X_all", "y_all"]:
            data[key] = np.array(data[key], dtype=np.float32)

        # ── HPT on VALIDATION ─────────────────────────────────────────────
        best_mae, best_params = np.inf, None
        for params in NBEATS_HPT_GRID:
            try:
                model = build_nbeats(DL_LOOKBACK, n_feat=data["n_feat"], **params)
                model.fit(
                    data["X_tr"], data["y_tr"],
                    validation_data=(data["X_vl"], data["y_vl"]),
                    epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                    callbacks=get_callbacks(), verbose=0
                )
                pv_sc  = model.predict(data["X_vl"], verbose=0).flatten()
                pv_inv = data["sc_y"].inverse_transform(pv_sc.reshape(-1, 1)).flatten()
                yv_inv = data["sc_y"].inverse_transform(data["y_vl"].reshape(-1, 1)).flatten()
                mae    = float(np.mean(np.abs(yv_inv - pv_inv)))
                if not np.isfinite(mae):
                    log.warning(f"     {params} → MAE_val=nan/inf (skipped)")
                    continue
                log.info(f"     {params} → MAE_val={mae:.4f}")
                if mae < best_mae:
                    best_mae, best_params = mae, params
            except Exception as e:
                log.warning(f"     {params} → failed: {e}")
                continue

        if best_params is None:
            log.error(f"  {tid}: no N-BEATS config converged. Skipping.")
            continue

        log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

        # ── Retrain on TRAIN+VAL, evaluate TEST ───────────────────────────
        # X_vl is already inside X_tv (train+val), so using validation_data=(X_vl,...)
        # would cause data leakage in early stopping. Monitor training loss only.
        X_tv = np.vstack([data["X_tr"], data["X_vl"]])
        y_tv = np.concatenate([data["y_tr"], data["y_vl"]])
        model_final = build_nbeats(DL_LOOKBACK, n_feat=data["n_feat"], **best_params)
        history = model_final.fit(
            X_tv, y_tv,
            callbacks=[
                EarlyStopping(monitor="loss", patience=DL_PATIENCE,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="loss", factor=0.5,
                                  patience=DL_PATIENCE // 2, min_lr=1e-6, verbose=0),
            ],
            epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, verbose=0
        )

        # ── Test evaluation ────────────────────────────────────────────────
        pt_sc  = model_final.predict(data["X_te"], verbose=0).flatten()
        pt_inv = data["sc_y"].inverse_transform(pt_sc.reshape(-1, 1)).flatten()
        yt_inv = data["sc_y"].inverse_transform(data["y_te"].reshape(-1, 1)).flatten()
        y_true_s = pd.Series(yt_inv, index=data["idx_te"])
        y_pred_s = pd.Series(pt_inv, index=data["idx_te"])
        metrics  = compute_metrics(y_true_s, y_pred_s)
        log.info(f"  ✓ TEST → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        # Validation forecast for 3-way plot
        pv2_sc  = model_final.predict(data["X_vl"], verbose=0).flatten()
        pv2_inv = data["sc_y"].inverse_transform(pv2_sc.reshape(-1, 1)).flatten()
        pred_vl_s = pd.Series(pv2_inv, index=data["idx_vl"])

        nome_modelo = f"NBEATS_{mode}"
        plot_forecast_3way(data["y_series"], pred_vl_s, y_pred_s,
                           tid, nome_modelo, OUTPUT_NBEATS, suffix="")
        plot_residuals(y_true_s, y_pred_s, tid, nome_modelo, OUTPUT_NBEATS)

        # Training loss curve
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.history["loss"], label="Training loss", color="#2563eb", lw=1.5)
        ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — {nome_modelo}: Training Loss (MSE)",
                     fontsize=13)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_NBEATS}/loss_{tid}_{nome_modelo}.svg",
                    dpi=150, bbox_inches="tight"); plt.close()

        # ── Future forecast ────────────────────────────────────────────────
        try:
            future_idx = pd.date_range(
                data["y_series"].index[-1] + pd.Timedelta(days=1),
                pd.to_datetime(FUTURE_END_DATE), freq="D"
            )
            fut_exog_sc = None
            if mode == "MV" and data["sc_exog"] is not None:
                cal_fut = adicionar_features_calendario(
                    pd.Series(np.nan, index=future_idx))
                if weather is not None:
                    for col in [c for c in data["sel_features"]
                                if c not in cal_fut.columns and c in weather.columns]:
                        cal_fut[col] = weather[col].mean()
                sel_fut = [c for c in data["sel_features"] if c in cal_fut.columns]
                fut_raw = cal_fut[sel_fut].values.astype(np.float32)
                n_exog  = data["n_feat"] - 1
                if fut_raw.shape[1] < n_exog:
                    pad = np.zeros((len(future_idx), n_exog - fut_raw.shape[1]),
                                   dtype=np.float32)
                    fut_raw = np.hstack([fut_raw, pad])
                fut_exog_sc = data["sc_exog"].transform(fut_raw[:, :n_exog])

            # Retrain on all data for future projection
            model_fut = build_nbeats(DL_LOOKBACK, n_feat=data["n_feat"], **best_params)
            model_fut.fit(
                data["X_all"], data["y_all"],
                epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                callbacks=[EarlyStopping(monitor="loss", patience=DL_PATIENCE,
                                         restore_best_weights=True, verbose=0)],
                verbose=0
            )

            preds_fut = recursive_predict_nbeats(
                model_fut, data["X_all"][-1], fut_exog_sc,
                data["sc_y"], len(future_idx)
            )
            fut_series = pd.Series(preds_fut, index=future_idx)

            # Bootstrap CI
            all_pred_inv = data["sc_y"].inverse_transform(
                model_fut.predict(data["X_all"], verbose=0).reshape(-1, 1)).flatten()
            all_y_inv = data["sc_y"].inverse_transform(
                data["y_all"].reshape(-1, 1)).flatten()
            resids = all_y_inv - all_pred_inv
            rng    = np.random.default_rng(RANDOM_STATE)
            boots  = np.array([preds_fut + rng.choice(resids, size=len(future_idx),
                                                       replace=True)
                               for _ in range(200)])
            plot_future_forecast(
                data["y_series"], fut_series,
                pd.Series(np.percentile(boots, 2.5,  axis=0), index=future_idx),
                pd.Series(np.percentile(boots, 97.5, axis=0), index=future_idx),
                tid, nome_modelo, OUTPUT_FUTURE
            )
        except Exception as e:
            log.warning(f"  Future forecast {nome_modelo} failed: {e}")

        resultados.append({
            "transformador": tid, "nome_plot": MAPA_PLOT.get(tid, tid),
            "modelo": "NBEATS", "abordagem": mode,
            "features_sel": str(data["sel_features"]),
            "n_features_sel": len(data["sel_features"]),
            "best_params": str(best_params), "MAE_val": round(best_mae, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
        })

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    weather = None
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Weather data not loaded: {e}")

    log.info("=== N-BEATS UNIVARIADO ===")
    res_uv = treinar_nbeats(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_NBEATS}/metricas_NBEATS_UV.csv", index=False)

    log.info("=== N-BEATS MULTIVARIADO ===")
    res_mv = treinar_nbeats(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_NBEATS}/metricas_NBEATS_MV.csv", index=False)

    res = pd.concat([res_uv, res_mv])
    res.to_csv(f"{OUTPUT_NBEATS}/metricas_NBEATS.csv", index=False)
    print(res[["transformador", "modelo", "abordagem", "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformador", "abordagem"]).to_string(index=False))


if __name__ == "__main__":
    main()
