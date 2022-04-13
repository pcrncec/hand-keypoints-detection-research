import tensorflow as tf
from hand_detection.models.conv_block import ConvBlock


class ResBlock(tf.keras.Model):
    def __init__(self, filters):
        super(ResBlock, self).__init__()

        self.conv1_a = ConvBlock(filters // 2, 1)
        self.conv2_a = ConvBlock(filters // 2, 3, strides=2)
        self.conv3_a = ConvBlock(filters, 1)
        self.avg_pool = tf.keras.layers.AvgPool2D(2, strides=2)
        self.conv1_b = ConvBlock(filters, 1)

    def call(self, input_tensor):
        a_out = self.conv1_a(input_tensor)
        a_out = self.conv2_a(a_out)
        a_out = self.conv3_a(a_out)
        b_out = self.avg_pool(input_tensor)
        b_out = self.conv1_b(b_out)
        out = tf.keras.layers.concatenate([a_out, b_out])
        return out
