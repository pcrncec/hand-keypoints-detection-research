from tensorflow.keras.layers import Layer, Conv2D
import tensorflow as tf


class AttentionAugmentedConv(Layer):
    def __init__(self, filters, kernel_size, nh, dv, dk):
        super(AttentionAugmentedConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.nh = nh
        self.dv = int(dv * self.filters)
        self.dk = int(dk * self.filters)

        assert self.nh > 0
        assert self.dk % self.nh == 0
        assert self.dv % self.nh == 0

    def build(self, input_shape):
        shape = input_shape[1]
        self.conv_out = Conv2D(self.filters - self.dv, kernel_size=self.kernel_size, padding='same')
        self.qkv_conv = Conv2D(2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = Conv2D(self.dv, kernel_size=1)
        self.key_rel_w = tf.Variable(tf.random.normal((2 * shape - 1, self.dk // self.nh)), trainable=True, name="aac_krw")
        self.key_rel_h = tf.Variable(tf.random.normal((2 * shape - 1, self.dk // self.nh)), trainable=True, name="aac_krh")
        super(AttentionAugmentedConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        conv_out = self.conv_out(inputs)
        attn_out = self._self_attention_2d(inputs)
        return tf.concat([conv_out, attn_out], axis=3)

    def _self_attention_2d(self, inputs):
        _, h, w, _ = inputs.shape
        dkh = self.dk // self.nh
        dvh = self.dv // self.nh
        flatten_hw = lambda x, d: tf.reshape(x, [-1, self.nh, h * w, d])

        kqv = self.qkv_conv(inputs)
        k, q, v = tf.split(kqv, [self.dk, self.dk, self.dv], axis=3)
        q *= dkh ** -0.5

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)
        rel_logits_h, rel_logits_w = self._relative_logits(q, h, w)
        logits += rel_logits_h
        logits += rel_logits_w

        weights = tf.nn.softmax(logits)
        attn_out = tf.matmul(weights, flatten_hw(v, dvh))
        attn_out = tf.reshape(attn_out, [-1, self.nh, h, w, dvh])
        attn_out = self._combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return attn_out

    def _split_heads(self, inputs):
        _, h, w, d = inputs.shape
        splitted = tf.reshape(inputs, [-1, h, w, self.nh, d // self.nh])
        return tf.transpose(splitted, [0, 3, 1, 2, 4])

    def _relative_logits(self, q, h, w):
        rel_logits_w = self._relative_logits_1d(q, self.key_rel_w, h, w, [0, 1, 2, 4, 3, 5])
        q_transposed = tf.transpose(q, [0, 1, 3, 2, 4])
        rel_logits_h = self._relative_logits_1d(q_transposed, self.key_rel_h, w, h, [0, 1, 4, 2, 5, 3])
        return rel_logits_w, rel_logits_h

    def _relative_logits_1d(self, q, rel_k, h, w, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = tf.reshape(rel_logits, [-1, self.nh * h, w, 2 * w - 1])
        rel_logits = self._rel_to_abs(rel_logits)
        rel_logits = tf.expand_dims(tf.reshape(rel_logits, [-1, self.nh, h, w, w]), axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        rel_logits = tf.reshape(rel_logits, [-1, self.nh, h * w, h * w])
        return rel_logits

    def _rel_to_abs(self, x):
        b, nh, l, _ = x.shape
        # col_pad = tf.zeros((b, nh, l, 1))
        col_pad = tf.zeros_like(x[..., 0:1])
        x = tf.concat([x, col_pad], axis=3)
        flat_x = tf.reshape(x, [-1, nh, l * 2 * l])
        # flat_pad = tf.zeros((b, nh, l - 1))
        flat_pad = tf.zeros_like(x[:, :, 0:-1, 0])
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
        final_x = tf.reshape(flat_x_padded, [-1, nh, l + 1, 2 * l - 1])
        final_x = final_x[:, :, :l, l - 1:]
        return final_x

    def _combine_heads_2d(self, inputs):
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
        nh, channels = transposed.shape[-2:]
        ret_shape = transposed.shape[:-2] + [nh * channels]
        ret_shape = [-1, ret_shape[1], ret_shape[2], ret_shape[3]]
        return tf.reshape(transposed, ret_shape)
