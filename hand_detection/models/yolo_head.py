import tensorflow as tf
from hand_detection.models.conv_block import ConvBlock


class YoloHead(tf.keras.Model):
    def __init__(self, anchors, num_classes):
        super(YoloHead, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.conv_256_1 = ConvBlock(256, 3)
        self.zero_pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.conv_256_2 = ConvBlock(256, 3, strides=2, padding='valid')
        self.conv_256_3 = ConvBlock(256, 1)
        self.conv_512_1 = ConvBlock(512, 3)

    def _conv_classes_anchors(self, inp, num_anchors_stage):
        x = tf.keras.layers.Conv2D(
            filters=num_anchors_stage * (self.num_classes + 5),
            kernel_size=1,
            padding="same")(inp)
        x = tf.keras.layers.Reshape(
            (x.shape[1], x.shape[2], num_anchors_stage, self.num_classes + 5))(x)
        return x

    def call(self, input_tensor):
        # input_1 shape: (26, 26, 256)
        # input_2 shape: (13, 13, 256)
        x = self.conv_256_1(input_tensor[0])
        out_1 = self._conv_classes_anchors(x, len(self.anchors[0]))
        x = self.zero_pad(input_tensor[0])
        x = self.conv_256_2(x)
        x = tf.keras.layers.concatenate([x, input_tensor[1]])
        x = self.conv_256_3(x)
        x = self.conv_512_1(x)
        out_2 = self._conv_classes_anchors(x, len(self.anchors[1]))
        return out_1, out_2
