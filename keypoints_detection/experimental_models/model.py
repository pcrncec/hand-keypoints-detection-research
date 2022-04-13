from common.inverted_res_block import DenseBlock, TransitionLayer
from common.mish import mish
from backbones import BackboneRGB, BackboneGrayscale
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, AveragePooling2D, Conv2D, Reshape, BatchNormalization, Activation
import tensorflow as tf


class KeypointsDetector(Model):
    def __init__(self):
        super(KeypointsDetector, self).__init__()
        self.rgb_backbone = BackboneRGB()
        self.grayscale_backbone = BackboneGrayscale()
        self.dense1 = DenseBlock(num_iter=4, augmented=False, growth_rate=20)
        self.trans1 = TransitionLayer(64)
        self.dense2 = DenseBlock(num_iter=4, augmented=True, growth_rate=20)
        self.trans2 = TransitionLayer(64)
        self.dense3 = DenseBlock(num_iter=4, augmented=True, growth_rate=40)
        self.trans3 = TransitionLayer(128)
        self.dense4 = DenseBlock(num_iter=6, augmented=True, growth_rate=40)
        self.trans4 = TransitionLayer(256)
        self.dense5 = DenseBlock(num_iter=1, augmented=True, growth_rate=80)
        self.bn = BatchNormalization()
        self.mish = Activation(mish)
        self.avg_pool = AveragePooling2D(strides=2)

        # output layers
        self.out_base_kp = Conv2D(filters=2, kernel_size=1, padding='same', activation='tanh')
        self.out_reshape_base = Reshape((1, 2), dtype=tf.float32)
        self.out_off1 = Conv2D(filters=10, kernel_size=1, padding='same', activation='tanh')
        self.out_reshape_off1 = Reshape((5, 2), dtype=tf.float32)
        self.out_off2 = Conv2D(filters=10, kernel_size=1, padding='same', activation='tanh')
        self.out_reshape_off2 = Reshape((5, 2), dtype=tf.float32)
        self.out_off3 = Conv2D(filters=10, kernel_size=1, padding='same', activation='tanh')
        self.out_reshape_off3 = Reshape((5, 2), dtype=tf.float32)
        self.out_off4 = Conv2D(filters=10, kernel_size=1, padding='same', activation='tanh')
        self.out_reshape_off4 = Reshape((5, 2), dtype=tf.float32)

    def call(self, inputs):
        rgb_input, grayscale_input = inputs
        rgb_backbone_out = self.rgb_backbone(rgb_input)
        grayscale_backbone_out = self.grayscale_backbone(grayscale_input)
        concat_out = concatenate([rgb_backbone_out, grayscale_backbone_out])
        x = self.dense1(concat_out)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        x = self.trans4(x)

        aa_block = self.dense5(x)
        aa_block = self.bn(aa_block)
        aa_block = self.mish(aa_block)
        aa_block = self.avg_pool(aa_block)

        out_base = self.out_base_kp(aa_block)
        out_base_r = self.out_reshape_base(out_base)

        concat_off1 = concatenate([aa_block, out_base])
        out_off1 = self.out_off1(concat_off1)
        out_off1_r = self.out_reshape_off1(out_off1)

        concat_off2 = concatenate([out_base, out_off1])
        out_off2 = self.out_off2(concat_off2)
        out_off2_r = self.out_reshape_off2(out_off2)

        concat_off3 = concatenate([out_base, out_off1, out_off2])
        out_off3 = self.out_off3(concat_off3)
        out_off3_r = self.out_reshape_off3(out_off3)

        concat_off4 = concatenate([out_base, out_off1, out_off2, out_off3])
        out_off4 = self.out_off4(concat_off4)
        out_off4_r = self.out_reshape_off4(out_off4)

        out = concatenate([out_base_r, out_off1_r, out_off2_r, out_off3_r, out_off4_r], axis=1)
        return out


def create_model(show_summary=True):
    rgb_input = Input((224, 224, 3))
    grayscale_input = Input((224, 224, 1))
    keypoints_detector_model = KeypointsDetector()
    model = Model([rgb_input, grayscale_input], keypoints_detector_model([rgb_input, grayscale_input])
    if show_summary:
        model.summary()
    return model
