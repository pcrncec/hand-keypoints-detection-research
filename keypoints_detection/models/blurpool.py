from tensorflow.keras.layers import Layer, DepthwiseConv2D
from tensorflow.keras.initializers import constant
import numpy as np


class BlurPool2D(Layer):
    def __init__(self):
        super(BlurPool2D, self).__init__()

    def build(self, input_shape):
        blurpool_filter = np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])
        blurpool_filter = blurpool_filter / np.sum(blurpool_filter)
        blurpool_filter = np.repeat(blurpool_filter, input_shape[-1])
        kernel_shape = (3, 3, input_shape[-1], 1)
        blurpool_filter = np.reshape(blurpool_filter, kernel_shape)
        blurpool_init = constant(blurpool_filter)
        self.d_wise_conv = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                                           depthwise_initializer=blurpool_init, trainable=False)
        super(BlurPool2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.d_wise_conv(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]
