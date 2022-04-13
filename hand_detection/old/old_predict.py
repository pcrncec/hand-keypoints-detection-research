from hand_detection.old.old_model_utils import get_prediction_model
from hand_detection.preprocess_data import preprocess_img
import hand_detection.utils as utils
import cv2
import numpy as np
import time


WEIGHTS_PATH_OLD = 'model_checkpoints/model_weights-112-0.29.hdf5'
MAX_BOXES = 4
NUM_CLASSES = 4
anchors = utils.get_anchors()
model = get_prediction_model(anchors, MAX_BOXES, NUM_CLASSES, WEIGHTS_PATH_OLD)
IMAGE_DIMS = (416, 416)
SCORE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.6
capture = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
while True:
    _, captured_img = capture.read()
    normalized_img = preprocess_img(captured_img, IMAGE_DIMS)
    normalized_img_batch = np.expand_dims(normalized_img, axis=0)
    prediction = model.predict(normalized_img_batch)

    boxes, scores, classes = utils.get_pred_boxes_and_scores(prediction, anchors, IMAGE_DIMS, NUM_CLASSES, MAX_BOXES, SCORE_THRESHOLD,
                                                             IOU_THRESHOLD)

    CLASSES_LABEL_MAP_PATH = "../classes_label_map.pbtxt"
    # image_to_show = utils.visualize_pred_boxes_and_classes(normalized_img * 255.,
    #                                                        classes_label_map_path=CLASSES_LABEL_MAP_PATH,
    #                                                        boxes=boxes.numpy(), scores=scores.numpy(),
    #                                                        classes=classes)
    if len(boxes) > 0:
        b = boxes[0].numpy()
        cv2.rectangle(normalized_img, (b[1], b[0]), (b[3], b[2]), (255, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = int(1/(new_frame_time - prev_frame_time))
        fps = str(fps)
        prev_frame_time = new_frame_time

        cv2.putText(normalized_img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        # cv2.imshow('Prediction', image_to_show / 255.)
        cv2.imshow('Prediction', normalized_img)
    if (cv2.waitKey(1) & 0XFF == ord("q")) or (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()
