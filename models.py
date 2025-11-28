###models/baseline_arima.py
import statsmodels.api as sm
import numpy as np

def arima_forecast(series, train_end_idx, steps=1, order=(5,1,0)):
train = series.iloc[:train_end_idx+1]
try:
model = sm.tsa.ARIMA(train, order=order).fit()
forecast = model.forecast(steps=steps)
return np.array(forecast)
except Exception as e:
#fallback : repeat last value
last = train.iloc[-1]
return np.array([last]*steps)

###models/lstm_model.py
from tensorflow import keras
from tensorflow.keras import layers

def build_standard_lstm(input_shape, units=64, dropout=0.2):
model = keras.Sequential([
layers.Input(shape=input_shape),
layers.LSTM(units, return_sequences=False),
layers.Dropout(dropout),
layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
return model

###models/attention_layer.py
import tensorflow as tf
from tensorflow.keras import layers


class SelfAttention(layers.Layer):
def __init__(self, units, **kwargs):
super(SelfAttention, self).__init__(**kwargs)
self.units = units

def build(self, input_shape):
d = input_shape[-1]
self.Wq = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', trainable=True)
self.Wk = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', trainable=True)
self.Wv = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', trainable=True)
super(SelfAttention, self).build(input_shape)

def call(self, inputs, return_attention=False):

Q = tf.tensordot(inputs, self.Wq, axes=1)
K = tf.tensordot(inputs, self.Wk, axes=1)
V = tf.tensordot(inputs, self.Wv, axes=1)
scores = tf.matmul(Q, K, transpose_b=True)
scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))
weights = tf.nn.softmax(scores, axis=-1)
context = tf.matmul(weights, V) # (batch, time, units)
context_vector = tf.reduce_mean(context, axis=1) # (batch, units)
if return_attention:
return context_vector, weights
return context_vector

def get_config(self):
return {'units': self.units}


### models/attention_lstm_model.py
from tensorflow import keras
from tensorflow.keras import layers
from .attention_layer import SelfAttention

def build_attention_lstm(input_shape, lstm_units=64, attn_units=32, dropout=0.2):
inputs = layers.Input(shape=input_shape)
x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
attn = SelfAttention(attn_units)
context = attn(x) # (batch, attn_units)
x = layers.Concatenate()([context])
x = layers.Dropout(dropout)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
return model, attn
