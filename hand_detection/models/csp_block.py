import tensorflow as tf
from hand_detection.models.conv_block import ConvBlock


class CSPBlock(tf.keras.Model):
    def __init__(self, filters):
        super(CSPBlock, self).__init__()

        self.conv1 = ConvBlock(filters, 3)
        self.conv2_1 = ConvBlock(filters // 2, 3)
        self.conv2_2 = ConvBlock(filters // 2, 3)
        self.conv3 = ConvBlock(filters, 1)

    def call(self, input_tensor):
        conv1_out = self.conv1(input_tensor)
        conv2_out1 = self.conv2_1(conv1_out)
        conv2_out2 = self.conv2_2(conv2_out1)
        concat_out1 = tf.keras.layers.concatenate([
            conv2_out1, conv2_out2
        ])
        conv3_out = self.conv3(concat_out1)
        concat_out2 = tf.keras.layers.concatenate([
            conv1_out, conv3_out
        ])
        return concat_out2

