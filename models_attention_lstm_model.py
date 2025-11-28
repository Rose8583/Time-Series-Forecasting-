from tensorflow.keras import layers, Model
from .attention_layer import SelfAttention

def build_attention_lstm(input_shape, lstm_units=64, attn_units=32, dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    attn = SelfAttention(attn_units)
    context, attn_weights = attn(x, return_attention=True)
    x = layers.Dropout(dropout)(context)
    outputs = layers.Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    attn_extractor = Model(inputs=inputs, outputs=attn_weights)
    return model, attn_extractor
