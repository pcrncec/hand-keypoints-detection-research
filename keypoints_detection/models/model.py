from attention_augmented_conv import AttentionAugmentedConv
from inverted_res_block import DenseBlock, TransitionLayer, mish
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, AveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf


def create_model(input_shape, show_summary=True):
    input_layer = Input(input_shape)
    dense_block1 = DenseBlock(num_iter=3, augmented=False)(input_layer)
    t1 = TransitionLayer(filters=128)(dense_block1)
    dense_block2 = DenseBlock(num_iter=3, augmented=False)(t1)
    t2 = TransitionLayer(filters=128)(dense_block2)
    dense_block3 = DenseBlock(num_iter=1, augmented=True)(t2)
    t3 = TransitionLayer(filters=64)(dense_block3)
    dense_block4 = DenseBlock(num_iter=8, augmented=True)(t3)
    t4 = TransitionLayer(filters=128)(dense_block4)
    dense_block5 = DenseBlock(num_iter=4, augmented=True)(t4)
    t5 = TransitionLayer(filters=128)(dense_block5)
    dense_block6 = DenseBlock(num_iter=4, augmented=True)(t5)
    t6 = TransitionLayer(filters=256)(dense_block6)
    dense_block7 = DenseBlock(num_iter=6, augmented=True)(t6)
    t7 = TransitionLayer(filters=256)(dense_block7)
    dense_block8 = DenseBlock(num_iter=8, augmented=True)(t7)

    aa_block = AttentionAugmentedConv(filters=100, kernel_size=3, nh=10, dv=0.1, dk=0.1)(dense_block8)
    aa_block = BatchNormalization()(aa_block)
    aa_block = Activation(mish)(aa_block)
    aa_block = AveragePooling2D(strides=2)(aa_block)
    final_layer = Conv2D(filters=42, kernel_size=1, padding='same', activation='relu')(aa_block)
    final_layer = Reshape((-1, 21, 2), dtype=tf.float32)(final_layer)

    model = Model(input_layer, final_layer)
    if show_summary:
        model.summary()
    return model
