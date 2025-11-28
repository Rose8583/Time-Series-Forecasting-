# models/attention_layer.py
import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    """
    Self-attention layer compatible with Keras functional API.

    Call signature:
      context = SelfAttention(units)(inputs)
    or
      context, weights = SelfAttention(units)(inputs, return_attention=True)

    Inputs:
      inputs: Tensor(shape=(batch, time, features))
    Outputs:
      context: Tensor(shape=(batch, units))  -- reduced across time
      weights (optional): Tensor(shape=(batch, time, time))  -- attention matrices
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        # input_shape: (batch, time, features)
        d = int(input_shape[-1])
        # Learned linear projections for Q, K, V
        self.Wq = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wq')
        self.Wk = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wk')
        self.Wv = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wv')
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        """
        inputs: (batch, time, features)
        return_attention: if True, also return attention weights
        """
        # Linear projections
        Q = tf.tensordot(inputs, self.Wq, axes=1)  # (batch, time, units)
        K = tf.tensordot(inputs, self.Wk, axes=1)  # (batch, time, units)
        V = tf.tensordot(inputs, self.Wv, axes=1)  # (batch, time, units)

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, time, time)
        scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))

        weights = tf.nn.softmax(scores, axis=-1)  # attention weights, row-normalized

        # Weighted values
        context_seq = tf.matmul(weights, V)  # (batch, time, units)

        # Reduce across time to produce a fixed-size context vector.
        # Using mean reduction is simple and effective; you can also use last timestep or weighted sum.
        context_vector = tf.reduce_mean(context_seq, axis=1)  # (batch, units)

        if return_attention:
            return context_vector, weights
        return context_vector

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
