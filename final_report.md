# Final Report — Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

## Project summary
This project compares three modeling approaches for multivariate time series forecasting:
- ARIMA (statistical baseline)
- Standard LSTM (deep recurrent baseline)
- Attention-enhanced LSTM (Bi-LSTM + custom Self-Attention layer)

The evaluation uses rolling-origin cross-validation and reports RMSE, MAE, and MAPE.

---

## Dataset
- Source: S&P 500 historical data via yfinance (`^GSPC`)
- Period: configurable; by default 2000-01-01 to present
- Features engineered:
  - LogClose (target)
  - Return (pct_change)
  - MA_5, MA_10 (moving averages)
  - Vol_5 (rolling volume mean)
- Records: > 500 after feature engineering

---

## Preprocessing
- Missing values dropped after rolling feature construction
- Log-transform applied to Close -> LogClose, used as primary target (stabilizes variance)
- MinMax scaling applied to all features (scaler fit on training data in a production setting; for simplicity fitted on full data in this pipeline — **recommendation:** fit scaler per training fold to avoid leakage)
- Windowing: sequence length (window) configurable (default 60 timesteps)
- Multi-step horizon: pred_horizon configurable (default 5); pipeline predicts last value at horizon for simplicity

---

## Models and hyperparameters
**ARIMA**
- Order: (5,1,0) by default
- Forecast uses univariate LogClose series

**Standard LSTM**
- LSTM units: 64
- Dropout: 0.2
- Dense output: 1
- Optimizer: Adam (default lr 0.001)
- Loss: MSE

**Attention-LSTM**
- Bidirectional LSTM (units 64) → sequence outputs
- SelfAttention layer: projected Q/K/V, scaled dot-product, softmax weight matrix
- Attention units: 32
- Output: context vector → Dropout → Dense(1)
- We provide an attention extractor model for interpretability (weights shape: batch x T x T)

Recommended hyperparameters (used in experiments):
- window size: 60
- pred_horizon: 5
- epochs: 20–50 depending on dataset size
- batch size: 32

---

## Cross-validation strategy
- Rolling-origin (time-series aware):
  - Start with an initial training window (configurable, default near 1000 sequences or 40% of samples)
  - Slide forward by a fixed step (configurable)
  - For each fold: train exclusively on earlier data and evaluate on the immediate next horizon
- Metrics are recorded per fold and then aggregated (mean ± std)

---

## Evaluation metrics
- RMSE (root mean squared error)
- MAE (mean absolute error)
- MAPE (mean absolute percentage error)

A template results table (replace with actual values from `outputs/performance_metrics.csv`):

| Model       | RMSE (mean ± std) | MAE (mean ± std) | MAPE (mean ± std) |
|-------------|-------------------:|------------------:|-------------------:|
| ARIMA       | 123.45 ± 12.3      | 98.76 ± 9.8       | 3.21% ± 0.45%      |
| LSTM        | 110.12 ± 11.0      | 85.23 ± 8.4       | 2.78% ± 0.37%      |
| Attn-LSTM   | 101.05 ± 9.7       | 78.90 ± 7.1       | 2.30% ± 0.30%      |

*(The numeric values above are placeholders — use `outputs/performance_metrics.csv` to fill real numbers.)*

---

## Attention mechanism implementation (summary)
- SelfAttention projects hidden states into Q/K/V, computes scaled dot-product scores, softmax-normalizes to get weights A, then computes context = A @ V and reduces across time to get a fixed context vector.
- The pipeline saves attention matrices per test fold (`outputs/attn_weights_fold{i}.npy`) for inspection.

---

## Interpretability / Insights
- Heatmaps of attention weights typically show:
  - Strong attention on recent timesteps (short-term importance)
  - Periodic peaks in many series (seasonality)
  - Low weights assigned to noisy/uninformative timesteps
- These patterns support qualitative reasoning about model behavior and can be used to explain predictions to stakeholders.

---

## Limitations & recommendations
- Current scaler is fit on full data for convenience — **fix** by fitting scaler only on train folds in production runs.
- ARIMA hyperparameters can be tuned (auto_arima) for better baseline fairness.
- Use teacher forcing or direct multi-step decoders if strict multi-step sequence forecasting (all steps) is required.
- Add K-fold parameter sweep (KerasTuner or Optuna) to robustly tune hyperparameters.

---

## How to reproduce results
1. `pip install -r requirements.txt`
2. `python main.py`
3. Open `outputs/performance_metrics.csv`, `outputs/summary.json`, and `outputs/attn_weights_fold*.npy`
4. Use `notebooks/analysis_notebook.ipynb` to visualize results and attention heatmaps.

