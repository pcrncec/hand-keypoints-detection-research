import tensorflow as tf
from hand_detection.models.conv_block import ConvBlock


class AuxiliaryResBlock(tf.keras.Model):
    def __init__(self, filters, use_strides):
        super(AuxiliaryResBlock, self).__init__()

        self.conv = ConvBlock(filters // 2, 3, strides=2) if use_strides else ConvBlock(filters // 2, 3)
        self.conv2 = ConvBlock(filters // 2, 3)
        self.ca = ChannelAttention(filters // 2)
        self.sa = SpatialAttention()

    def call(self, input_tensor):
        conv1_out = self.conv(input_tensor)
        conv2_out = self.conv2(conv1_out)
        ca_out = self.ca(conv2_out) * conv2_out
        sa_out = self.sa(ca_out) * ca_out
        out = tf.keras.layers.concatenate([
            sa_out, conv1_out
        ])
        return out


class ChannelAttention(tf.keras.Model):
    def __init__(self, filters):
        super(ChannelAttention, self).__init__()

        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=1)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=1)
        self.fc = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters, 1, use_bias=False),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters, 1, use_bias=False)
        ])
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, input_tensor):
        avg_pool_out = self.fc(self.avg_pool(input_tensor))
        max_pool_out = self.fc(self.max_pool(input_tensor))
        out = avg_pool_out + max_pool_out
        return self.sigmoid(out)


class SpatialAttention(tf.keras.Model):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=7,
                                           padding='same',
                                           use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, input_tensor):
        avg_out = tf.reduce_mean(input_tensor, axis=3, keepdims=True)
        assert avg_out.shape[-1] == 1
        max_out = tf.reduce_max(input_tensor, axis=3, keepdims=True)
        assert max_out.shape[-1] == 1
        concat_out = tf.concat([avg_out, max_out], axis=3)
        assert concat_out.shape[-1] == 2
        out = self.conv(concat_out)
        return self.sigmoid(out)

