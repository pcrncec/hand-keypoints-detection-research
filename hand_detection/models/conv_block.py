import tensorflow as tf


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, padding='same'):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        return self.leaky_relu(x)
