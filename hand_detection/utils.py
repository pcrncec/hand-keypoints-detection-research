import os
import random
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


def get_anchors():
    return np.array([[[np.divide(10, 13), np.divide(14, 13)],
                      [np.divide(23, 13), np.divide(27, 13)], [np.divide(37, 13), np.divide(58, 13)]],
                     [[np.divide(81, 26), np.divide(82, 26)], [np.divide(135, 26), np.divide(169, 26)],
                      [np.divide(344, 26), np.divide(319, 26)]]])


def avg_IOU(anns, centroids):
    n, d = anns.shape
    s = 0.
    for i in range(anns.shape[0]):
        s += max(anchors_iou(anns[i], centroids))
    return s / n


def anchors_iou(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)
    return np.array(similarities)


def kmeans(annotation_dims, num_anchors):
    ann_num = annotation_dims.shape[0]
    prev_assignments = np.ones(ann_num) * (-1)
    iteration = 0
    indices = [random.randrange(annotation_dims.shape[0]) for _ in range(num_anchors)]
    centroids = annotation_dims[indices]
    anchor_dim = annotation_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - anchors_iou(annotation_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        centroid_sums = np.zeros((num_anchors, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += annotation_dims[i]
        for j in range(num_anchors):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()


def generate_anchors(resized_res, images_dir_path, all_bboxes, num_anchors, num_pool_layers):
    width, height = resized_res
    downsampling_factor = np.power(2, num_pool_layers)
    assert width % downsampling_factor == 0
    assert height % downsampling_factor == 0
    grid_height = height // downsampling_factor
    grid_width = width // downsampling_factor
    img_paths = os.listdir(images_dir_path)
    annotation_dims = []
    for img_path in img_paths:
        full_img_path = os.path.join(images_dir_path, img_path)
        img = cv2.imread(full_img_path)
        img_height, img_width = img.shape[:2]
        img_bboxes = all_bboxes.get(img_path)
        cell_width = img_width / grid_width
        cell_height = img_height / grid_height
        for img_bbox in img_bboxes:
            relative_width = np.divide(img_bbox[2], cell_width)
            relative_height = np.divide(img_bbox[3], cell_height)
            annotation_dims.append([relative_width, relative_height])
    annotation_dims = np.array(annotation_dims)
    return kmeans(annotation_dims, num_anchors)


def decode_conv_output(num_classes, conv_output, anchors):
    conv_shape = conv_output.shape
    conv_dims = conv_shape[1:3]
    num_anchors = len(anchors)
    conv_output = tf.reshape(conv_output, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    anchors_tensor = tf.reshape(tf.Variable(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
    conv_height_index = tf.range(start=0, limit=conv_dims[0])
    conv_width_index = tf.range(start=0, limit=conv_dims[1])
    conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])
    conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = tf.reshape(tf.transpose(conv_width_index), [-1])
    conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
    conv_index = tf.cast(tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2]), tf.float32)
    reshaped_conv_dim = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), tf.float32)

    box_xy_center = tf.sigmoid(conv_output[..., :2])
    box_xy_center = (box_xy_center + conv_index) / reshaped_conv_dim
    box_wh = tf.exp(conv_output[..., 2:4])
    box_wh = box_wh * anchors_tensor / reshaped_conv_dim
    box_conf = tf.sigmoid(conv_output[..., 4:5])
    box_class_prob = tf.nn.softmax(conv_output[..., 5:])
    return box_xy_center, box_wh, box_conf, box_class_prob


def filter_boxes(boxes, conf, box_class_probs, threshold):
    box_scores = conf * box_class_probs
    classes = tf.argmax(box_scores, axis=-1)
    scores = tf.reduce_max(box_scores, axis=-1)
    prediction_mask = scores >= threshold
    filtered_boxes = tf.boolean_mask(boxes, prediction_mask)
    filtered_scores = tf.boolean_mask(scores, prediction_mask)
    filtered_classes = tf.boolean_mask(classes,  prediction_mask)
    return filtered_boxes, filtered_scores, filtered_classes


def rescale_boxes(boxes, image_dims):
    image_dims_tensor = tf.stack([image_dims[0], image_dims[1], image_dims[0], image_dims[1]])
    image_dims_tensor = tf.cast(tf.reshape(image_dims_tensor, [1, 4]), tf.float32)
    rescaled_boxes = boxes * image_dims_tensor
    return rescaled_boxes


def get_pred_boxes_and_scores(conv_outputs, anchors, image_dims, num_classes, max_boxes, score_thresh, iou_thresh):
    boxes, scores, classes = [], [], []
    for conv_out, anchor in zip(conv_outputs, anchors):
        xy_center, wh, conf, class_prob = decode_conv_output(num_classes, conv_out, anchor)
        box_mins = xy_center - (wh / 2.)
        box_maxes = xy_center + (wh / 2.)
        concat_boxes = np.concatenate(
            [box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]], axis=-1)
        filtered_boxes, filtered_scores, filtered_classes = filter_boxes(concat_boxes[0], conf[0], class_prob[0], threshold=score_thresh)
        filtered_boxes = rescale_boxes(filtered_boxes, image_dims)
        boxes.append(filtered_boxes)
        scores.append(filtered_scores)
        classes.append(filtered_classes)
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes = np.concatenate(classes)

    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_thresh)
    boxes = tf.gather(boxes, nms_indices)
    scores = tf.gather(scores, nms_indices)
    classes = tf.gather(classes, nms_indices)
    return boxes, scores, classes


def visualize_pred_boxes_and_classes(image, classes_label_map_path, boxes, scores, classes):
    category_index = label_map_util.create_category_index_from_labelmap(classes_label_map_path, use_display_name=True)
    img = viz_utils.visualize_boxes_and_labels_on_image_array(image=image, boxes=boxes,
                                                              classes=classes,
                                                              category_index=category_index,
                                                              scores=scores,
                                                              use_normalized_coordinates=True)
    return img


def crop_hand(img, box, img_dims=(416, 416)):
    y_min, x_min, y_max, x_max = box
    adj_x_min = np.max(0, x_min - 50)
    adj_x_max = np.min(img_dims[0], x_max + 50)
    adj_y_min = np.max(0, y_min - 50)
    adj_y_max = np.min(img_dims[1], y_max + 50)
    cropped_img = img[adj_y_min:adj_y_max, adj_x_min:adj_x_max]
    return cropped_img
