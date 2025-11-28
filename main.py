arima_pred_log=arima_forecast(aligned_logclose, train_end_idx=train_idx.stop - 1, steps=len(y_test))
arima_pred_price=np.exp(arima_pred_log)

##Standard LSTM
model_lstm = build_standard_lstm(input_shape=X_train.shape[1:])
val_split = max(1, int(0.1 * len(X_train)))
X_tr, y_tr = X_train[:-val_split], y_train[:-val_split]
X_val, y_val = X_train[-val_split:], y_train[-val_split:]
model_lstm.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, verbose=2)
pred_lstm_scaled = model_lstm.predict(X_test).flatten()

##invert scaling for predictions (only first column was target scaled)
dummy=np.zeros((len(pred_lstm_scaled), values.shape[1]))
dummy[:,0]=pred_lstm_scaled
inv_lstm_log=scaler.inverse_transform(dummy)[:,0]
lstm_pred_price=np.exp(inv_lstm_log)

##Attention LSTM
model_attn, attn_layer=build_attention_lstm(input_shape=X_train.shape[1:],lstm_units=64, attn_units=32)
model_attn.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, verbose=2)
pred_attn_scaled=model_attn.predict(X_test).flatten()
dummy[:,0]=pred_attn_scaled
inv_attn_log=scaler.inverse_transform(dummy)[:,0]
attn_pred_price=np.exp(inv_attn_log)

##actual prices
actual_log=df_feat['LogClose'].loc[dates[test_idx]].values
actual_price=np.exp(actual_log)

##metrics
for name, preds in [('ARIMA', arima_pred_price), ('LSTM', lstm_pred_price), ('Attn-LSTM', attn_pred_price)]:
r=rmse(actual_price, preds)
a=mae(actual_price, preds)
m=mape(actual_price, preds)
records.append({'fold': i+1, 'model': name, 'rmse': float(r), 'mae': float(a), 'mape': float(m)})

df_metrics=pd.DataFrame(records)
df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False)

##simple report
summary=df_metrics.groupby('model').mean()[['rmse','mae','mape']].round(6).to_dict()
report={
'params':{'seq_len': seq_len, 'pred_horizon': pred_horizon, 'epochs': epochs},
'summary': summary
}
with open(os.path.join(OUTPUT_DIR, 'analysis.txt'), 'w') as f:
f.write(json.dumps(report, indent=2))
print('Completed. Outputs saved to', OUTPUT_DIR)

if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=str, default='2000-01-01')
parser.add_argument('--end', type=str, default=None)
args = parser.parse_args()
run_pipeline(start=args.start, end=args.end)
