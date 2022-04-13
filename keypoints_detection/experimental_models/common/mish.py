import tensorflow as tf


def mish(x):
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))
