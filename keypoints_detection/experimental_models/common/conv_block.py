from mish import mish
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation, Layer
import tensorflow as tf


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, activation=None, strides=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.kernel_size, self.strides, self.padding)
        self.bn = BatchNormalization()
        if self.activation_name is None:
            self.activation = None
        else:
            assert self.activation_name == 'leaky_relu' or self.activation_name == 'mish'
            if self.activation_name == 'leaky_relu':
                self.activation = LeakyReLU()
            else:
                self.activation = Activation(mish, dtype=tf.float32)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
