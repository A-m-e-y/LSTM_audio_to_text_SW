import tensorflow as tf
import numpy as np

class CustomLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomLSTMCell, self).__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Create weights for input and hidden state separately
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units * 4,), initializer='zeros', name='bias')

    def sw_dot(self, x, w):
        """Software dot product using NumPy for now."""
        # x: (batch_size, input_dim)
        # w: (input_dim, units*4)
        return tf.linalg.matmul(x, w)

    def call(self, inputs, states):
        h_tm1, c_tm1 = states  # previous hidden and cell states

        # Calculate z = x * Wx + h_tm1 * Wh + b
        z = self.sw_dot(inputs, self.kernel) + self.sw_dot(h_tm1, self.recurrent_kernel) + self.bias

        # Split z into 4 parts for gates
        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)

        # Gates
        i = tf.sigmoid(z0)          # Input gate
        f = tf.sigmoid(z1)          # Forget gate
        g = tf.tanh(z2)             # Candidate memory
        o = tf.sigmoid(z3)          # Output gate

        # New cell state and hidden state
        c = f * c_tm1 + i * g
        h = o * tf.tanh(c)

        return h, [h, c]
