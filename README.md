# Nikkei 225 Volatility Forecasting: Structural Ensembling of HAR-RV and Temporal Fusion Transformers

This repository implements an advanced multi-horizon volatility forecasting framework for the Nikkei 225 index. It addresses the limitations of purely autoregressive neural networks in handling heavy-tailed financial data and cross-asset macroeconomic shocks.

By introducing a **Structural Ensemble** that orthogonalizes the signal—delegating linear volatility persistence to an econometric Log-HAR-RV model, and non-linear asymmetry to a Temporal Fusion Transformer (TFT)—the model achieves a highly competitive out-of-sample MAPE.

##  Methodology & Architecture

The architecture bridges classical financial econometrics with attention-based Deep Learning:
1. **Econometric Baseline (Log-HAR-RV)**: Captures the multi-scale deterministic persistence of the market (daily, weekly, monthly components).
2. **Deep Learning Correction (TFT)**: A Temporal Fusion Transformer models the residual non-linearities, optimizing the `QuantileLoss` to accurately estimate tail risks (10th and 90th percentiles).
3. **Variable Selection Networks (VSN)**: Dynamically filters noisy exogenous features, activating specific cross-asset variables only during market regime shifts.

##  Key Features & Domain Engineering

The dataset spans from 2007 to 2026, incorporating the 2008 Great Financial Crisis and the recent Yen carry-trade unwinding shocks. Key engineered features include:
* **Microstructural Asymmetry**: `signed_volume` and `down_volume` (Panic Selling Indicator) to capture the leverage effect.
* **Global Spillover & FX**: S&P 500 returns, VIX levels, and USD/JPY dynamics.
* **Japanese Market Specifics**: Future-known calendar covariates, including exact days remaining to the **Special Quotation (SQ)** technical expirations in Tokyo.
* **Exact Window Dropout**: A deterministic feature calculating the exact absolute return dropping out of the rolling window at $t+k$.

##  Results & Benchmarks

The Structural Ensemble was benchmarked on a strictly out-of-sample test set against a Naive Random Walk, a standalone Log-HAR-RV, and previous research utilizing Temporal Kolmogorov-Arnold Networks (TKAN).

| Model Architecture | Out-of-Sample MAPE (t+1) | 5-Day Horizon QLIKE |
| :--- | :---: | :---: |
| TKAN (Endogenous Baseline) | ~4.30% | - |
| Naive Random Walk | 4.26% | 0.0697 |
| Log-HAR-RV (Linear) | 4.62% | 0.0575 |
| Pure TFT | 3.91% | 0.0562 |
| **Structural Ensemble (HAR + TFT)** | **3.81%** | **0.0536** |

**Key Findings via VSN Interpretability:**
* The model correctly delegates the baseline inertia to the HAR-RV anchor.
* During normal market regimes, USD/JPY returns hold a standard weight (~6.5%). During high-volatility stress events, the VSN dynamically doubles this weight to ~11.8%, perfectly capturing the mechanics of Yen carry-trade liquidations.
