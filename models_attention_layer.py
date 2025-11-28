import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        self.Wq = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wq')
        self.Wk = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wk')
        self.Wv = self.add_weight(shape=(d, self.units), initializer='glorot_uniform', name='Wv')
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        Q = tf.tensordot(inputs, self.Wq, axes=1)
        K = tf.tensordot(inputs, self.Wk, axes=1)
        V = tf.tensordot(inputs, self.Wv, axes=1)
        scores = tf.matmul(Q, K, transpose_b=True)
        scores /= tf.sqrt(tf.cast(self.units, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)
        context_vector = tf.reduce_mean(context, axis=1)
        if return_attention:
            return context_vector, weights
        return context_vector

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
