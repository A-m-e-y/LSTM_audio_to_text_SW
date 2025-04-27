import tensorflow as tf
import numpy as np
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CustomLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units * 4), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units * 4,), initializer='zeros', name='bias')

    def sw_dot(self, x, w):
        """Software dot product."""
        return tf.linalg.matmul(x, w)

    def call(self, inputs, states):
        h_tm1, c_tm1 = states

        z = self.sw_dot(inputs, self.kernel) + self.sw_dot(h_tm1, self.recurrent_kernel) + self.bias

        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        g = tf.tanh(z2)
        o = tf.sigmoid(z3)

        c = f * c_tm1 + i * g
        h = o * tf.tanh(c)

        return h, [h, c]
