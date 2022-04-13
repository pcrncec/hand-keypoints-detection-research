from preprocess_data import HandLocalizationData
from model_utils import create_model
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def train(images_dir_path, boxes_file_path, batch_size, num_epochs):
    train_data = HandLocalizationData(images_dir_path, boxes_file_path)

    model = create_model(train_data.anchors, train_data.max_boxes, num_classes=4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    STEPS_PER_EPOCH = train_data.num_data // batch_size
    train_data_gen = train_data.generator(batch_size, num_epochs)
    tf.config.run_functions_eagerly(True)

    terminate_nan_cb = TerminateOnNaN()
    ckpt_filepath = 'model_checkpoints/model_weights-{epoch:02d}-{loss:.2f}.hdf5'
    model_ckpt_cb = ModelCheckpoint(filepath=ckpt_filepath, monitor='loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=8, verbose=1, min_lr=1e-7)

    CKPT_FILENAME = 'model_checkpoints/model_weights-59-0.31.hdf5'
    model.load_weights(CKPT_FILENAME)

    VAL_DIR_PATH = 'images/HandLocalization/EgoHands/valid/images/'
    VAL_BOXES_PATH = 'val_boxes.npz'
    val_data = HandLocalizationData(VAL_DIR_PATH, VAL_BOXES_PATH, max_boxes=train_data.max_boxes)
    VAL_STEPS = val_data.num_data // batch_size
    val_data_gen = val_data.generator(batch_size, num_epochs)
    model.fit(train_data_gen, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=val_data_gen, validation_steps=VAL_STEPS,
              callbacks=[terminate_nan_cb, model_ckpt_cb, reduce_lr])
    print('Training completed.')


IMAGES_DIR_PATH = 'images/HandLocalization/EgoHands/train/images/'
BOXES_FILE_PATH = 'training_boxes.npz'
BATCH_SIZE = 16
NUM_EPOCHS = 500
train(IMAGES_DIR_PATH, BOXES_FILE_PATH, BATCH_SIZE, NUM_EPOCHS)
