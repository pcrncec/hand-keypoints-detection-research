import os
from experimental_models.model import create_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
import tensorflow as tf
from generator import data_generator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def loss_fn(y_true, y_pred):
    rmse = tf.sqrt(tf.keras.losses.MSE(y_true, y_pred))
    loss = tf.concat([rmse[..., :11], tf.multiply(rmse[..., 11:], 2)], axis=-1)
    return loss


def train(train_images_path, keypoints_path, num_epochs, batch_size, lr=1e-3, img_res=(224, 224), initial_epoch=0):
    data_gen = data_generator(train_images_path, keypoints_path, num_epochs, batch_size, img_res)
    steps_per_epoch = len(os.listdir(train_images_path)) // batch_size
    model = create_model()

    optimizer = Adam(learning_rate=lr)

    reduce_lr_cb = ReduceLROnPlateau(monitor='loss', patience=4, min_lr=1e-6, factor=0.1)
    ckpt_path = 'C:/Users/Patrik/Desktop/airsynth_checkpoints/keypoints/keypoints_model_offset_weights-{epoch:02d}-{loss:.4f}.hdf5'
    ckpt_cb = ModelCheckpoint(ckpt_path, monitor='loss', save_best_only=True, save_weights_only=True)

    mixed_precision.set_global_policy('mixed_float16')

    weights_path = 'C:/Users/Patrik/Desktop/airsynth_checkpoints/keypoints/keypoints_model_offset_weights-44-0.0171.hdf5'
    model.load_weights(weights_path)

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(data_gen, batch_size=batch_size, epochs=num_epochs, callbacks=[reduce_lr_cb, ckpt_cb],
              steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch)


TRAIN_IMAGES_PATH = 'C:/Users/Patrik/Desktop/airsynth_data/keypoints/Keypoints_FreiHAND/training/rgb'
KEYPOINTS_PATH = 'C:/Users/Patrik/Desktop/airsynth_data/keypoints/train_2d_keypoints_offsets.npz'
NUM_EPOCHS = 60
BATCH_SIZE = 4
train(TRAIN_IMAGES_PATH, KEYPOINTS_PATH, NUM_EPOCHS, BATCH_SIZE, lr=3e-6, initial_epoch=44)
