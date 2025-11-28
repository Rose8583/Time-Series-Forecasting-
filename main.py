#!/usr/bin/env python3
"""
main.py
Full pipeline:
 - Fetch S&P500 (yfinance) or load local CSV
 - Feature engineering, scaling, sequence creation
 - Rolling-origin CV
 - Train ARIMA baseline, standard LSTM, attention-LSTM
 - Save metrics (CSV), attention weights (npy), and example plots
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ML / DL
import yfinance as yf
import statsmodels.api as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# local modules (if you split into utils/models these imports can change)
# Here we implement everything inline to keep single-file runnable.

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

########################
# Utility metrics
########################
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

########################
# Data acquisition & features
########################
def fetch_sp500(ticker="^GSPC", start="2000-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("Could not fetch data from yfinance.")
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df

def engineer_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['LogClose'] = np.log(df['Close'])
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Vol_5'] = df['Volume'].rolling(5).mean()
    df = df.dropna()
    return df

def create_sequences(values, target_col_index=0, seq_len=60, pred_horizon=1):
    X, y = [], []
    for i in range(seq_len, len(values) - pred_horizon + 1):
        X.append(values[i-seq_len:i])
        y.append(values[i+pred_horizon-1, target_col_index])
    return np.array(X), np.array(y)

########################
# Attention components
########################
import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.Wq = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wq')
        self.Wk = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wk')
        self.Wv = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wv')
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        # inputs: (batch, time, features)
        Q = tf.tensordot(inputs, self.Wq, axes=1)
        K = tf.tensordot(inputs, self.Wk, axes=1)
        V = tf.tensordot(inputs, self.Wv, axes=1)
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, time, time)
        scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)  # (batch, time, units)
        context_vector = tf.reduce_mean(context, axis=1)  # (batch, units)
        if return_attention:
            return context_vector, weights
        return context_vector

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

########################
# Models
########################
def build_standard_lstm(input_shape, units=64, dropout=0.2):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_attention_lstm(input_shape, lstm_units=64, attn_units=32, dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    attn = SelfAttention(attn_units)
    context, attn_weights = attn(x, return_attention=True)
    x = layers.Dropout(dropout)(context)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    attn_extractor = keras.Model(inputs=inputs, outputs=attn_weights)
    return model, attn_extractor

########################
# ARIMA helper
########################
def arima_forecast(series, train_end_idx, steps=1, order=(5,1,0)):
    # series: pandas Series aligned to sequence y dates
    train = series.iloc[:train_end_idx+1]
    try:
        model = sm.tsa.ARIMA(train, order=order).fit()
        fc = model.forecast(steps=steps)
        return np.array(fc)
    except Exception as e:
        last = train.iloc[-1]
        return np.array([last]*steps)

########################
# Rolling-origin CV and main pipeline
########################
def rolling_origin_splits(n_samples, initial_train, step, horizon, max_folds=10):
    folds = []
    start = initial_train
    fold = 0
    while start + horizon <= n_samples and fold < max_folds:
        train_slice = slice(0, start)
        test_slice = slice(start, start + horizon)
        folds.append((train_slice, test_slice))
        start += step
        fold += 1
    return folds

def invert_scaled_target(scaled_vals, scaler, n_features):
    # scaled_vals: (n,) values representing target scaled between 0-1
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:,0] = scaled_vals
    inv = scaler.inverse_transform(dummy)[:,0]
    return inv

def run_pipeline(start='2000-01-01', end=None, seq_len=60, pred_horizon=5,
                 initial_train_size=1000, step=180, max_folds=6, epochs=20, batch_size=32):
    print("Fetching data...")
    df = fetch_sp500(start=start, end=end)
    df_feat = engineer_features(df)
    features = ['LogClose','Return','MA_5','MA_10','Vol_5']
    values = df_feat[features].values
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)
    X_all, y_all = create_sequences(values_scaled, target_col_index=0, seq_len=seq_len, pred_horizon=pred_horizon)
    dates = df_feat.index[seq_len: seq_len + len(y_all)]

    n = len(X_all)
    initial_train = max( int(initial_train_size - seq_len), int(n*0.4) )
    folds = rolling_origin_splits(n, initial_train, step, pred_horizon, max_folds=max_folds)
    print(f"Total folds: {len(folds)} (n_samples={n})")
    recs = []

    for i, (train_slice, test_slice) in enumerate(folds):
        print(f"Fold {i+1}: train {train_slice.stop} test {test_slice.start}-{test_slice.stop}")
        X_train, y_train = X_all[train_slice], y_all[train_slice]
        X_test, y_test = X_all[test_slice], y_all[test_slice]

        # Build aligned original logclose series for ARIMA
        aligned_logclose = df_feat['LogClose'].loc[dates]

        # ARIMA forecast (works on original log scale)
        arima_pred_log = arima_forecast(aligned_logclose, train_end_idx=train_slice.stop - 1, steps=len(y_test))
        arima_pred_price = np.exp(arima_pred_log)

        # Standard LSTM
        model_lstm = build_standard_lstm(input_shape=X_train.shape[1:], units=64)
        val_split = max(1, int(0.1 * len(X_train)))
        X_tr, y_tr = X_train[:-val_split], y_train[:-val_split]
        X_val, y_val = X_train[-val_split:], y_train[-val_split:]
        model_lstm.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, verbose=0)
        pred_lstm_scaled = model_lstm.predict(X_test).flatten()
        inv_lstm_log = invert_scaled_target(pred_lstm_scaled, scaler, n_features=values.shape[1])
        lstm_pred_price = np.exp(inv_lstm_log)

        # Attention LSTM
        model_attn, attn_extractor = build_attention_lstm(input_shape=X_train.shape[1:], lstm_units=64, attn_units=32)
        model_attn.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, verbose=0)
        pred_attn_scaled = model_attn.predict(X_test).flatten()
        inv_attn_log = invert_scaled_target(pred_attn_scaled, scaler, n_features=values.shape[1])
        attn_pred_price = np.exp(inv_attn_log)

        # actual
        actual_log = df_feat['LogClose'].loc[dates[test_slice]].values
        actual_price = np.exp(actual_log)

        # metrics
        for name, preds in [('ARIMA', arima_pred_price), ('LSTM', lstm_pred_price), ('Attn-LSTM', attn_pred_price)]:
            r = rmse(actual_price, preds)
            a = mae(actual_price, preds)
            m = mape(actual_price, preds)
            recs.append({'fold': i+1, 'model': name, 'rmse': r, 'mae': a, 'mape': m})

        # save attention weights for this fold
        try:
            attn_weights = attn_extractor.predict(X_test)  # shape (batch, T, T)
            np.save(OUTPUT_DIR / f"attn_weights_fold{i+1}.npy", attn_weights)
        except Exception as e:
            print("Attention extractor failed:", e)

    df_metrics = pd.DataFrame(recs)
    df_metrics.to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)
    print("Saved:", OUTPUT_DIR / "performance_metrics.csv")
    # aggregated summary
    summary = df_metrics.groupby('model').agg(['mean','std'])[['rmse','mae','mape']]
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        f.write(summary.round(6).to_json())

    # Save human-readable report snippet
    with open(OUTPUT_DIR / "report_summary.txt", "w") as f:
        f.write("Aggregated results (mean ± std):\n")
        f.write(summary.round(6).to_string())

    print("Pipeline complete. Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    run_pipeline()
