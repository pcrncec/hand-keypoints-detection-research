from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, Activation
from blurpool import BlurPool2D
from attention_augmented_conv import AttentionAugmentedConv
from mish import mish
import tensorflow as tf


class InvertedResidualLayer(Layer):
    def __init__(self, augmented, growth_rate=10, e_factor=4, nh=4):
        super(InvertedResidualLayer, self).__init__()
        self.augmented = augmented
        self.growth_rate = growth_rate
        self.e_factor = e_factor
        self.nh = nh
        self.conv1 = Conv2D(self.growth_rate * self.e_factor, kernel_size=1, padding='same')
        self.bn1 = BatchNormalization()
        self.d_wise_conv = DepthwiseConv2D(kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(self.growth_rate, kernel_size=1, padding='same')
        self.bn3 = BatchNormalization()
        if augmented:
            self.attn_aug_block = self._attn_aug_block()
        else:
            self.attn_aug_block = None

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = Activation(mish)(x)
        x = self.d_wise_conv(x)
        x = self.bn2(x)
        x = Activation(mish)(x)
        if self.augmented:
            attn_aug_block_out = self.attn_aug_block(x)
            attn_aug_block_out = Activation(mish)(attn_aug_block_out)
            x = tf.concat([attn_aug_block_out, x], axis=-1)
        x = self.conv2(x)
        x = self.bn3(x)
        x = Activation(mish, dtype=tf.float32)(x)
        return x

    def _attn_aug_block(self):
        attn_aug_conv = AttentionAugmentedConv(self.growth_rate * self.e_factor, kernel_size=3,
                                               nh=self.nh, dv=0.1, dk=0.1)
        return tf.keras.Sequential([attn_aug_conv, BatchNormalization()])


class DenseBlock(Layer):
    def __init__(self, num_iter, augmented, growth_rate=10, e_factor=4, nh=4):
        super(DenseBlock, self).__init__()
        self.inverted_residuals = []
        for i in range(num_iter):
            ir = InvertedResidualLayer(augmented, growth_rate, e_factor, nh)
            self.inverted_residuals.append(ir)

    def call(self, inputs, **kwargs):
        x = [inputs]
        for layer in self.inverted_residuals:
            x.append(layer(tf.concat(x, axis=-1)))
        return tf.concat(x, axis=-1)


class TransitionLayer(Layer):
    def __init__(self, filters):
        super(TransitionLayer, self).__init__()
        self.conv = Conv2D(filters, kernel_size=1, padding='same')
        self.blur_pool = BlurPool2D()
        self.bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.blur_pool(x)
        x = self.bn(x)
        return x
