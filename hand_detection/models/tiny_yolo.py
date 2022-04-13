import tensorflow as tf
from hand_detection.models.auxiliary_resblock import AuxiliaryResBlock
from hand_detection.models.conv_block import ConvBlock
from hand_detection.models.csp_block import CSPBlock
from hand_detection.models.res_block import ResBlock


class TinyYolo(tf.keras.Model):
    def __init__(self, anchors, num_classes):
        super(TinyYolo, self).__init__()

        self.conv_32_3x3 = ConvBlock(32, 3, strides=2)
        self.conv_64_3x3 = ConvBlock(64, 3, strides=2)
        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(128)
        self.auxiliary_res_block_128 = AuxiliaryResBlock(128, use_strides=False)
        self.auxiliary_res_block_256 = AuxiliaryResBlock(256, use_strides=True)
        self.csp_block = CSPBlock(256)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv_512_3x3_1 = ConvBlock(512, 3)
        self.conv_512_3x3_2 = ConvBlock(512, 3)
        self.conv_512_3x3_3 = ConvBlock(512, 3)
        self.conv_128_1x1 = tf.keras.layers.Conv2D(128, 1)
        self.upsampling = tf.keras.layers.Conv2DTranspose(256, 1, 2)
        self.conv_256_3x3 = ConvBlock(256, 3)
        self.conv_256_1x1_1 = tf.keras.layers.Conv2D(256, 1)
        self.conv_256_1x1_2 = tf.keras.layers.Conv2D(256, 1)
        self.final_conv_1 = tf.keras.layers.Conv2D(filters=len(anchors[0]) * (num_classes + 5),
                                                   kernel_size=1, padding="same")
        self.final_conv_2 = tf.keras.layers.Conv2D(filters=len(anchors[1]) * (num_classes + 5),
                                                   kernel_size=1, padding="same")

    def call(self, input_tensor):
        conv_out = self.conv_32_3x3(input_tensor)
        conv_out = self.conv_64_3x3(conv_out)
        res_block_out1 = self.res_block1(conv_out)
        auxiliary_res_block_out1 = self.auxiliary_res_block_128(res_block_out1)
        auxiliary_res_block_out2 = self.auxiliary_res_block_256(auxiliary_res_block_out1)
        concat_out1 = tf.keras.layers.concatenate([
            res_block_out1, auxiliary_res_block_out1
        ])
        res_block_out2 = self.res_block2(concat_out1)
        concat_out2 = tf.keras.layers.concatenate([
            res_block_out2, auxiliary_res_block_out2
        ])
        csp_block_out1 = self.csp_block(concat_out2)
        csp_block_out1 = self.max_pool(csp_block_out1)
        conv_out = self.conv_512_3x3_1(csp_block_out1)
        conv_out = self.conv_512_3x3_2(conv_out)
        upsample_conv_out = self.conv_128_1x1(conv_out)
        upsample_conv_out = self.upsampling(upsample_conv_out)
        concat_out3 = tf.keras.layers.concatenate([
            res_block_out2, upsample_conv_out
        ])
        out1 = self.conv_256_3x3(concat_out3)
        out1 = self.conv_256_1x1_1(out1)
        out1 = self.final_conv_1(out1)
        out2 = self.conv_512_3x3_3(conv_out)
        out2 = self.conv_256_1x1_2(out2)
        out2 = self.final_conv_2(out2)
        return out1, out2
