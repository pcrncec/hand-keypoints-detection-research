from experimental_models.model import create_model
from generator import preprocess_img
import cv2
import numpy as np
import time
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def plot_hand(img, coords):
    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]

    print("COORDS:", coords)
    for i, coord in enumerate(coords):
        cv2.circle(img, (coord[0], coord[1]), 4, colors[i], thickness=2)


def postprocess_keypoints(keypoints, img_res=224):
    keypoints *= img_res
    base_kp = keypoints[0:1]
    offset_1 = keypoints[1:6] + base_kp
    offset_2 = keypoints[6:11] + offset_1
    offset_3 = keypoints[11:16] + offset_2
    offset_4 = keypoints[16:] + offset_3
    return np.concatenate((base_kp, offset_1, offset_2, offset_3, offset_4))


WEIGHTS_PATH = 'C:/Users/Patrik/Desktop/airsynth_checkpoints/keypoints/keypoints_model_offset_weights-43-0.0173.hdf5'
IMAGE_DIMS = (224, 224, 3)
model = create_model(show_summary=False)
model.load_weights(WEIGHTS_PATH)
capture = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while True:
    _, captured_img = capture.read()
    normalized_img = preprocess_img(captured_img, IMAGE_DIMS[:2])
    normalized_img_batch = np.expand_dims(normalized_img, axis=0)

    prediction = model.predict(normalized_img_batch)[0]
    pred_keypoints = postprocess_keypoints(prediction)
    plot_hand(normalized_img, pred_keypoints)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    fps = str(fps)
    prev_frame_time = new_frame_time

    cv2.putText(normalized_img, fps, (7, 50), font, 2, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Prediction', normalized_img)
    if (cv2.waitKey(1) & 0XFF == ord("q")) or (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()
