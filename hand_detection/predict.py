from preprocess_data import preprocess_img
import cv2
import numpy as np
import time
import tensorflow as tf
import random
import colorsys

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

WEIGHTS_PATH = 'yolov4-tiny-v2'
MAX_BOXES = 2
NUM_CLASSES = 4
model = tf.keras.models.load_model(WEIGHTS_PATH)
# saved_model_loaded = tf.saved_model.load(WEIGHTS_PATH, tags=[tag_constants.SERVING])
# infer = saved_model_loaded.signatures['serving_default']

IMAGE_DIMS = (416, 416)
SCORE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6
capture = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0


def draw_bbox(image, bboxes, classes, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        font_scale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, font_scale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


all_fps = []


@tf.function
def pred(input_img):
    return model(input_img)


while True:
    _, captured_img = capture.read()
    if captured_img is not None:
        normalized_img = preprocess_img(captured_img, IMAGE_DIMS)
        normalized_img_batch = np.expand_dims(normalized_img, axis=0)
        input_img = tf.constant(normalized_img_batch, dtype=tf.float32)

        # pred_bbox = infer(batch_data)
        predictions = pred(input_img)[0]
        # predictions = model.predict(input_img)[0]
        # pred(input_img)

        if len(predictions) > 0:
            boxes = []
            pred_conf = []
            for prediction in predictions:
                boxes.append(prediction[..., 0:4])
                pred_conf.append(prediction[..., 4:])
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=MAX_BOXES,
                max_total_size=MAX_BOXES,
                iou_threshold=IOU_THRESHOLD,
                score_threshold=SCORE_THRESHOLD
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            src_classes = {0: 'myleft', 1: 'myright', 2: 'yourleft', 3: 'yourright'}
            image = draw_bbox(normalized_img, pred_bbox, src_classes)

        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        all_fps.append(fps)
        print(np.average(all_fps))
        fps = str(fps)
        prev_frame_time = new_frame_time

        cv2.putText(normalized_img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Prediction', normalized_img)
    if (cv2.waitKey(1) & 0XFF == ord("q")) or (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()
