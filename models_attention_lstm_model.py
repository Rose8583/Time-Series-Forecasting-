# models/attention_lstm_model.py
from tensorflow import keras
from tensorflow.keras import layers
from .attention_layer import SelfAttention

def build_attention_lstm(input_shape, lstm_units=64, attn_units=32, dropout=0.2):
    """
    Builds a Bi-LSTM followed by a SelfAttention layer.
    Returns:
      - model: Keras Model (inputs -> prediction)
      - attn_extractor: Keras Model (same inputs -> attention weights)
    """
    inputs = layers.Input(shape=input_shape, name="input_sequence")
    # Bidirectional LSTM that returns sequences (one vector per timestep)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name="bilstm")(inputs)

    # Self-Attention: returns context vector and attention weights when return_attention=True
    attn_layer = SelfAttention(attn_units)
    context, attn_weights = attn_layer(x, return_attention=True)

    x = layers.Dropout(dropout, name="dropout")(context)
    outputs = layers.Dense(1, name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="attention_lstm_model")
    model.compile(optimizer='adam', loss='mse')

    # Attention extractor (for interpretability): inputs -> attention weight matrices
    attn_extractor = keras.Model(inputs=inputs, outputs=attn_weights, name="attn_extractor")

    return model, attn_extractor
