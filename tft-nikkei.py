import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Fix for PyTorch >= 2.6 regarding Lightning checkpoint loading
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

pl.seed_everything(42, workers=True)


# ==============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ==============================================================================
def load_and_prepare_data(
    use_csv: bool = False,
    csv_path: str = "nikkei_data.csv",
    vol_window: int = 20
) -> pd.DataFrame:
    if use_csv:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
    else:
        tickers = {
            "N225": "^N225",      # Nikkei 225
            "SP500": "^GSPC",     # S&P 500 (US Spillover)
            "VIX": "^VIX",        # VIX (Implied Volatility proxy)
            "USDJPY": "JPY=X"     # USD/JPY Exchange Rate
        }
        raw_data = {}
        for name, ticker in tickers.items():
            # Extended to 2007 to include the 2008 Great Financial Crisis
            data = yf.download(
                ticker,
                start="2007-01-01",
                end="2026-01-01",
                auto_adjust=False,
                progress=False
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data.sort_index(inplace=True)
            raw_data[name] = data

        df = pd.DataFrame(index=raw_data["N225"].index)
        df["N225_Open"] = raw_data["N225"]["Open"]
        df["N225_High"] = raw_data["N225"]["High"]
        df["N225_Low"] = raw_data["N225"]["Low"]
        df["N225_Close"] = raw_data["N225"]["Close"]

        vol_series = raw_data["N225"]["Volume"].replace(0, np.nan).ffill()
        df["N225_Volume"] = np.log1p(vol_series)

        # Forward-fill only to align US/Japan holidays without data leakage
        df["SP500_Close"] = raw_data["SP500"]["Close"].reindex(df.index).ffill()
        df["VIX_Close"] = raw_data["VIX"]["Close"].reindex(df.index).ffill()
        df["USDJPY_Close"] = raw_data["USDJPY"]["Close"].reindex(df.index).ffill()
        df.dropna(inplace=True)

    # --- Log-Returns & Leverage Effect ---
    df["n225_ret"] = np.log(df["N225_Close"] / df["N225_Close"].shift(1))
    df["n225_ret_sq"] = df["n225_ret"] ** 2
    df["n225_neg_ret"] = np.minimum(df["n225_ret"], 0.0)

    # [KEY INSIGHT 1]: MICROSTRUCTURE & PANIC SELLING
    # Standard volume is noisy. "down_volume" strictly isolates the selling pressure
    # during market crashes, acting as a powerful non-linear trigger for volatility.
    df["signed_volume"] = df["N225_Volume"] * np.sign(df["n225_ret"])
    df["down_volume"] = np.where(df["n225_ret"] < 0, df["N225_Volume"], 0.0)

    df["sp500_ret"] = np.log(df["SP500_Close"] / df["SP500_Close"].shift(1))
    df["usdjpy_ret"] = np.log(df["USDJPY_Close"] / df["USDJPY_Close"].shift(1))

    # --- Garman-Klass Intraday Estimator ---
    log_hl = np.log(df["N225_High"] / df["N225_Low"])
    log_co = np.log(df["N225_Close"] / df["N225_Open"])
    gk_var = np.maximum(0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2), 1e-8)
    df["gk_vol_daily"] = np.sqrt(gk_var) * np.sqrt(252)

    # --- Target & Multi-Scale HAR Components (Corsi 2009) ---
    df["realized_vol"] = (df["n225_ret"].rolling(window=vol_window).std() * np.sqrt(252)).clip(lower=1e-4)
    df["rv_5d"] = (df["n225_ret"].rolling(window=5).std() * np.sqrt(252)).clip(lower=1e-4)
    df["rv_10d"] = (df["n225_ret"].rolling(window=10).std() * np.sqrt(252)).clip(lower=1e-4)
    df["ewma_vol"] = (np.sqrt(df["n225_ret_sq"].ewm(alpha=0.06).mean()) * np.sqrt(252)).clip(lower=1e-4)
    df["gk_vol_10d"] = df["gk_vol_daily"].rolling(window=10).mean().clip(lower=1e-4)

    # [KEY INSIGHT 2]: DETERMINISTIC WINDOW EXIT
    # At time t, we already know which return will drop out of the 20-day rolling window
    # at time t+1, t+2, etc. Providing this exactly removes lagging uncertainty.
    df["dropping_ret_abs"] = np.abs(df["n225_ret"].shift(vol_window))

    # --- Exogenous Factors ---
    df["usdjpy_vol"] = (df["usdjpy_ret"].rolling(window=vol_window).std() * np.sqrt(252)).clip(lower=1e-4)
    df["vix_level"] = (df["VIX_Close"] / 100.0).clip(lower=1e-4)

    # --- Calendar & Technical Expirations (Special Quotation - SQ) ---
    df["day_of_week"] = df.index.dayofweek.astype(str)
    df["month"] = df.index.month.astype(str)

    def days_to_sq(date: pd.Timestamp) -> float:
        """Calculates days left to the second Friday of the month (Japanese SQ day)."""
        first_day = date.replace(day=1)
        days_until_friday = (4 - first_day.weekday() + 7) % 7
        second_friday = first_day + pd.Timedelta(days=days_until_friday + 7)
        if date > second_friday:
            next_month = (first_day + pd.Timedelta(days=32)).replace(day=1)
            days_until_friday = (4 - next_month.weekday() + 7) % 7
            second_friday = next_month + pd.Timedelta(days=days_until_friday + 7)
        return float((second_friday - date).days)

    df["days_to_sq"] = [days_to_sq(d) for d in df.index]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df = df.reset_index()
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["series_id"] = "NIKKEI225"

    return df


# ==============================================================================
# 2. STAGE 1: ECONOMETRIC LINEAR HAR-RV BASELINE (NO DATA LEAKAGE)
# ==============================================================================
def fit_and_predict_har(
    df: pd.DataFrame,
    train_cutoff: int,
    max_horizon: int = 5
) -> tuple[pd.DataFrame, dict]:
    """
    Fits an asymmetric Log-HAR-RV model ONLY on the Training Set using OLS.
    Returns the 1-step ahead prediction as 'har_anchor' and the multi-step predictions.
    """
    X_cols = ["log_rv_20", "log_rv_10", "log_rv_5", "log_ewma", "log_gk", "n225_neg_ret", "dropping_ret_abs"]

    work_df = pd.DataFrame(index=df.index)
    work_df["log_rv_20"] = np.log(df["realized_vol"])
    work_df["log_rv_10"] = np.log(df["rv_10d"])
    work_df["log_rv_5"] = np.log(df["rv_5d"])
    work_df["log_ewma"] = np.log(df["ewma_vol"])
    work_df["log_gk"] = np.log(df["gk_vol_10d"])
    work_df["n225_neg_ret"] = df["n225_neg_ret"]
    work_df["dropping_ret_abs"] = df["dropping_ret_abs"]

    X_all = np.column_stack([np.ones(len(df)), work_df[X_cols].values])
    y_log_all = np.log(df["realized_vol"].values)

    har_models = {}
    har_preds_by_h = {}

    for h in range(1, max_horizon + 1):
        # Strict temporal alignment: predict y_{t+h} using info up to t
        valid_train_idx = np.arange(0, train_cutoff - h + 1)
        X_train = X_all[valid_train_idx]
        y_train = y_log_all[valid_train_idx + h]

        # OLS Estimation
        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        har_models[h] = beta

        # Jensen's inequality correction for inverse log transformation
        residuals_train = y_train - (X_train @ beta)
        jensen_corr = 0.5 * np.var(residuals_train)

        pred_log_h = (X_all @ beta) + jensen_corr
        har_preds_by_h[h] = np.exp(pred_log_h).clip(min=1e-4)

    # Provide 1-step prediction made at t-1 to time t (Zero leakage)
    df["har_anchor"] = np.roll(har_preds_by_h[1], shift=1)
    df.loc[0, "har_anchor"] = df.loc[0, "realized_vol"]

    return df, har_preds_by_h


# ==============================================================================
# 3. EVALUATION METRICS (QLIKE & PERCENTAGE ERRORS)
# ==============================================================================
def compute_volatility_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    eps = 1e-4
    y_t = np.maximum(y_true, eps)
    y_p = np.maximum(y_pred, eps)

    mse = float(np.mean((y_p - y_t) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_p - y_t)))

    mape = float(np.mean(np.abs(y_p - y_t) / y_t) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_p - y_t) / (np.abs(y_t) + np.abs(y_p))) * 100.0)
    wape = float((np.sum(np.abs(y_p - y_t)) / np.sum(y_t)) * 100.0)

    # QLIKE: The gold standard for volatility evaluation (Patton, 2011)
    var_t = y_t ** 2
    var_p = y_p ** 2
    qlike = float(np.mean((var_t / var_p) - np.log(var_t / var_p) - 1.0))

    print(f"\n--- {label} ---")
    print(f"MSE   : {mse:.6f}  | RMSE  : {rmse * 100:.2f}% (vol points) | MAE  : {mae * 100:.2f}% (vol points)")
    print(f"MAPE  : {mape:.2f}%     | sMAPE : {smape:.2f}%             | WAPE : {wape:.2f}%")
    print(f"QLIKE : {qlike:.6f}")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "WAPE": wape, "QLIKE": qlike}


def extract_aligned_har_predictions(raw_x: dict, har_preds_by_h: dict, max_horizon: int = 5) -> np.ndarray:
    """Aligns HAR predictions exactly with the TFT dataloader indices."""
    decoder_time_idx = raw_x["decoder_time_idx"].detach().cpu().numpy()
    t_origin_idx = decoder_time_idx[:, 0] - 1  # Base time t

    n_samples = len(t_origin_idx)
    har_matrix = np.zeros((n_samples, max_horizon), dtype=np.float32)
    for h in range(1, max_horizon + 1):
        har_matrix[:, h - 1] = har_preds_by_h[h][t_origin_idx]
    return har_matrix


# ==============================================================================
# 4. MAIN PIPELINE: TFT TRAINING & STRUCTURAL ENSEMBLE
# ==============================================================================
if __name__ == "__main__":
    df = load_and_prepare_data(use_csv=False, vol_window=20)

    max_encoder_length = 30
    max_prediction_length = 5

    total_len = len(df)
    train_cutoff = int(total_len * 0.80)
    val_cutoff = int(total_len * 0.90)

    # Stage 1: Calibrate HAR-RV model strictly on Training Set
    df, har_preds_by_h = fit_and_predict_har(df, train_cutoff=train_cutoff, max_horizon=max_prediction_length)

    unknown_reals = [
        "realized_vol",
        "har_anchor",       # Linear structural anchor passed to the network
        "ewma_vol",
        "rv_5d",
        "rv_10d",
        "gk_vol_daily",
        "gk_vol_10d",
        "n225_ret",
        "n225_neg_ret",
        "signed_volume",    
        "down_volume",      
        "sp500_ret",
        "vix_level",
        "usdjpy_ret",
        "usdjpy_vol",
    ]

    training_dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="realized_vol",
        group_ids=["series_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_known_reals=["days_to_sq", "dropping_ret_abs"],
        time_varying_unknown_reals=unknown_reals,
        categorical_encoders={
            "day_of_week": NaNLabelEncoder(add_nan=True),
            "month": NaNLabelEncoder(add_nan=True),
        },
        # [KEY INSIGHT 3]: LOCAL SCALING (NON-STATIONARITY FIX)
        # Normalizes each 30-day window independently in log-space.
        # The model only has to predict relative % changes, surviving market regime shifts.
        target_normalizer=EncoderNormalizer(transformation="log"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df[df["time_idx"] <= val_cutoff],
        min_prediction_idx=train_cutoff + 1,
        stop_randomization=True,
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        min_prediction_idx=val_cutoff + 1,
        stop_randomization=True,
    )

    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=max(len(validation_dataset), 1), num_workers=0)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=max(len(test_dataset), 1), num_workers=0)

    ckpt_dir = "checkpoints_tft_ensemble"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=10, verbose=False, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=ckpt_dir, filename="tft-ensemble-{epoch:02d}", save_top_k=1, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        logger=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=5e-4,
        hidden_size=24,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=12,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )

    print(f"\nStarting TFT Training | Total Parameters: {tft.size() / 1e3:.1f}k")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # ==============================================================================
    # 5. ENSEMBLE WEIGHTS CALIBRATION (ON VALIDATION SET)
    # ==============================================================================
    best_model_path = checkpoint_callback.best_model_path
    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, weights_only=False)
    except TypeError:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    val_raw = best_tft.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="auto", logger=False))
    val_tft_median = val_raw.output.prediction[:, :, 3].detach().cpu().numpy()
    val_actuals = val_raw.x["decoder_target"].detach().cpu().numpy()
    val_har_preds = extract_aligned_har_predictions(val_raw.x, har_preds_by_h, max_prediction_length)

    # Optimize convex combination weights (alpha) to blend TFT and HAR-RV
    optimal_alphas = np.zeros(max_prediction_length)
    for h_idx in range(max_prediction_length):
        def loss_func(alpha):
            blend = alpha * val_tft_median[:, h_idx] + (1.0 - alpha) * val_har_preds[:, h_idx]
            return np.mean(np.abs(blend - val_actuals[:, h_idx]) / np.maximum(val_actuals[:, h_idx], 1e-4))

        res = minimize_scalar(loss_func, bounds=(0.0, 1.0), method="bounded")
        optimal_alphas[h_idx] = res.x

    print("\n--- Optimal Ensemble Weights (Calibrated on Validation Set) ---")
    for h_idx, alpha in enumerate(optimal_alphas, start=1):
        print(f"Horizon t+{h_idx} -> TFT Weight: {alpha * 100:5.1f}% | HAR-RV Weight: {(1 - alpha) * 100:5.1f}%")

    # ==============================================================================
    # 6. OUT-OF-SAMPLE EVALUATION (TEST SET)
    # ==============================================================================
    raw_predictions = best_tft.predict(
        test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="auto", logger=False)
    )

    predictions = raw_predictions.output.prediction
    tft_median = predictions[:, :, 3].detach().cpu().numpy()
    tft_q10 = predictions[:, :, 1].detach().cpu().numpy()
    tft_q90 = predictions[:, :, 5].detach().cpu().numpy()
    actuals = raw_predictions.x["decoder_target"].detach().cpu().numpy()

    encoder_targets = raw_predictions.x["encoder_target"].detach().cpu().numpy()
    naive_pred_t1 = encoder_targets[:, -1]
    naive_pred_5d = np.repeat(naive_pred_t1[:, None], max_prediction_length, axis=1)

    har_test_preds = extract_aligned_har_predictions(raw_predictions.x, har_preds_by_h, max_prediction_length)

    # [KEY INSIGHT 4]: STRUCTURAL ENSEMBLE
    # Merges linear econometrics (perfect for baseline persistence) with Deep Learning
    # (perfect for non-linear shocks, VIX, JPY, and microstructures).
    ensemble_median = optimal_alphas * tft_median + (1.0 - optimal_alphas) * har_test_preds
    
    # Coherent quantile shifting around the new Ensemble median
    shift_t1 = ensemble_median[:, 0] - tft_median[:, 0]
    ensemble_q10_t1 = np.maximum(tft_q10[:, 0] + shift_t1, 1e-4)
    ensemble_q90_t1 = np.maximum(tft_q90[:, 0] + shift_t1, 1e-4)

    print("\n======================================================================")
    print("               OUT-OF-SAMPLE COMPARISON: STEP t+1                     ")
    print("======================================================================")
    compute_volatility_metrics(actuals[:, 0], naive_pred_t1, "1. Naive Baseline (Random Walk t+1)")
    compute_volatility_metrics(actuals[:, 0], har_test_preds[:, 0], "2. Log-HAR-RV Econometric Model (t+1)")
    compute_volatility_metrics(actuals[:, 0], tft_median[:, 0], "3. Pure TFT with Anchor (t+1)")
    compute_volatility_metrics(actuals[:, 0], ensemble_median[:, 0], "4. STRUCTURAL ENSEMBLE (HAR + TFT) [t+1]")

    print("\n======================================================================")
    print("            OUT-OF-SAMPLE COMPARISON: 5-DAY HORIZON                   ")
    print("======================================================================")
    compute_volatility_metrics(actuals, naive_pred_5d, "1. Naive Baseline (5-Day Horizon)")
    compute_volatility_metrics(actuals, har_test_preds, "2. Log-HAR-RV Econometric Model (5-Day Horizon)")
    compute_volatility_metrics(actuals, tft_median, "3. Pure TFT with Anchor (5-Day Horizon)")
    compute_volatility_metrics(actuals, ensemble_median, "4. STRUCTURAL ENSEMBLE (HAR + TFT) [5-Day Horizon]")

    # ==============================================================================
    # 7. VSN VARIABLES IMPORTANCE
    # ==============================================================================
    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    encoder_vars = best_tft.encoder_variables
    encoder_weights = interpretation["encoder_variables"].detach().cpu().numpy()
    encoder_weights_pct = (encoder_weights / (encoder_weights.sum() + 1e-12)) * 100.0

    exogenous_list = ["sp500_ret", "vix_level", "usdjpy_ret", "usdjpy_vol", "signed_volume", "down_volume"]

    print("\n--- Global Encoder Variables Ranking (VSN Weights) ---")
    for var, weight in sorted(zip(encoder_vars, encoder_weights_pct), key=lambda x: x[1], reverse=True):
        tipo = "[EXOGENOUS/MICRO]" if var in exogenous_list else "[ENDOGENOUS/HAR]"
        print(f"{var:<20} {tipo:<18}: {weight:6.2f}%")

    decoder_vars = best_tft.decoder_variables
    decoder_weights = interpretation["decoder_variables"].detach().cpu().numpy()
    decoder_weights_pct = (decoder_weights / (decoder_weights.sum() + 1e-12)) * 100.0

    # ==============================================================================
    # 8. VISUALIZATION DASHBOARD (4 QUADRANTS)
    # ==============================================================================
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Full Test Set (Actual vs Structural Ensemble)
    ax1 = axes[0, 0]
    ax1.plot(actuals[:, 0], label="Actual Volatility (Target)", color="#2ca02c", linewidth=1.5)
    ax1.plot(ensemble_median[:, 0], label="Structural Ensemble (HAR + TFT)", color="#1f77b4", linewidth=1.5, alpha=0.9)
    ax1.fill_between(
        range(len(actuals)), ensemble_q10_t1, ensemble_q90_t1,
        color="#1f77b4", alpha=0.2, label="10%-90% Confidence Interval"
    )
    ax1.set_title("Out-of-Sample Forecast: Structural Ensemble (Step t+1)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Annualized Volatility")
    ax1.set_xlabel("Days in Test Set")
    ax1.legend()

    # Plot 2: Zoom on the last 120 days (Naive vs TFT vs Ensemble)
    ax2 = axes[0, 1]
    zoom = min(120, len(actuals))
    ax2.plot(actuals[-zoom:, 0], label="Actual", color="#2ca02c", marker="o", markersize=3.5, linewidth=1.5)
    ax2.plot(tft_median[-zoom:, 0], label="Pure TFT", color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.plot(ensemble_median[-zoom:, 0], label="Ensemble (HAR+TFT)", color="#1f77b4", linewidth=1.8)
    ax2.fill_between(range(zoom), ensemble_q10_t1[-zoom:], ensemble_q90_t1[-zoom:], color="#1f77b4", alpha=0.2)
    ax2.set_title(f"Zoom on Last {zoom} Days: Tail Risk & Shock Handling", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Volatility")
    ax2.set_xlabel("Days")
    ax2.legend()

    # Plot 3: Encoder Variable Importance (VSN)
    ax3 = axes[1, 0]
    enc_sorted = sorted(zip(encoder_vars, encoder_weights_pct), key=lambda x: x[1])
    ax3.barh([x[0] for x in enc_sorted], [x[1] for x in enc_sorted], color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax3.set_title("Encoder Variable Importance (VSN)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Average Dynamic Weight (%)")

    # Plot 4: Blending Weights (TFT vs HAR) over Horizons (t+1..t+5)
    ax4 = axes[1, 1]
    horizons = [f"t+{h}" for h in range(1, max_prediction_length + 1)]
    ax4.bar(horizons, optimal_alphas * 100, label="TFT Weight (Non-Linear / Exogenous)", color="#1f77b4", alpha=0.85)
    ax4.bar(
        horizons, (1 - optimal_alphas) * 100, bottom=optimal_alphas * 100,
        label="Log-HAR-RV Weight (Linear Persistence)", color="#7f7f7f", alpha=0.6
    )
    ax4.set_title("Structural Decomposition: TFT vs HAR Weights by Horizon", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Percentage Contribution (%)")
    ax4.set_ylim(0, 100)
    ax4.legend()

    plt.tight_layout(pad=3.0)
    plt.show()