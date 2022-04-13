from common.conv_block import ConvBlock
from common.auxiliary_resblock import AuxiliaryResBlock
from common.inverted_res_block import DenseBlock, TransitionLayer
from tensorflow.keras.layers import Layer, concatenate, AveragePooling2D


class BackboneRGB(Layer):
    def __init__(self):
        super(BackboneRGB, self).__init__()

        self.conv_block1 = ConvBlock(32, 3, activation='mish')
        self.conv_block2 = ConvBlock(64, 3, activation='mish')
        self.trans1 = TransitionLayer(64)
        self.dense1 = DenseBlock(num_iter=4, augmented=False)
        self.trans2 = TransitionLayer(64)
        self.arb = AuxiliaryResBlock(128, use_strides=True)
        self.conv_block3 = ConvBlock(256, 3, activation='mish')
        self.avg_pool = AveragePooling2D()

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.trans1(x)
        x = self.dense1(x)
        trans_out = self.trans2(x)
        arb_out = self.arb(x)
        concat_out = concatenate([trans_out, arb_out])
        out = self.conv_block3(concat_out)
        out = self.avg_pool(out)
        return out


class BackboneGrayscale(Layer):
    def __init__(self):
        super(BackboneGrayscale, self).__init__()

        self.conv_block1 = ConvBlock(16, 3, activation='mish')
        self.conv_block2 = ConvBlock(32, 3, activation='mish')
        self.trans1 = TransitionLayer(32)
        self.dense1 = DenseBlock(num_iter=2, augmented=False)
        self.trans2 = TransitionLayer(32)
        self.dense2 = DenseBlock(num_iter=2, augmented=True)
        self.trans3 = TransitionLayer(64)

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.trans1(x)
        x = self.dense1(x)
        x = self.trans2(x)
        x = self.dense2(x)
        x = self.trans3(x)
        return x
