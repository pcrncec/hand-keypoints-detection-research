import numpy as np
import os
import cv2
import random
from utils import get_anchors


def normalize_bboxes(bboxes, img_width, img_height):
    bboxes[:, (0, 2)] = np.divide(bboxes[:, (0, 2)], img_width)
    bboxes[:, (1, 3)] = np.divide(bboxes[:, (1, 3)], img_height)
    return bboxes


def preprocess_img(img, resized_res, normalize=True):
    resized_img = cv2.resize(img, resized_res, interpolation=cv2.INTER_NEAREST)
    img_arr = np.array(resized_img, dtype=np.uint8, copy=False)
    if normalize:
        img_arr = img_arr.astype(np.float) / 255.
    return img_arr


def preprocess_bboxes(bboxes, anchors, img_res, num_pool_layers):
    width, height = img_res
    num_anchors = len(anchors)
    downsampling_factor = np.power(2, num_pool_layers)
    assert width % downsampling_factor == 0
    assert height % downsampling_factor == 0
    grid_height = height // downsampling_factor
    grid_width = width // downsampling_factor
    num_box_params = bboxes.shape[1]
    detectors_mask = np.zeros((grid_height, grid_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros((grid_height, grid_width, num_anchors, num_box_params), dtype=np.float32)
    for bbox in bboxes:
        bbox_classes = bbox[4:5]
        bbox = bbox[0:4] * np.array([grid_height, grid_width,
                                grid_height, grid_width])
        x_cell = np.minimum(np.floor(bbox[0]), grid_width - 1).astype(np.uint8)
        y_cell = np.minimum(np.floor(bbox[1]), grid_height - 1).astype(np.uint8)
        best_iou = 0
        best_anchor = 0
        for i, anchor in enumerate(anchors):
            bbox_maxes = bbox[2:4] / 2.
            bbox_mins = -bbox_maxes
            anchor_maxes = anchor / 2.
            anchor_mins = -anchor_maxes
            intersect_mins = np.maximum(bbox_mins, anchor_mins)
            intersect_maxes = np.minimum(bbox_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            bbox_area = bbox[2] * bbox[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (bbox_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = i
        if best_iou > 0:
            detectors_mask[y_cell, x_cell, best_anchor] = 1
            adj_bbox = np.array(
                [
                    bbox[0] - x_cell, bbox[1] - y_cell,
                    np.log(bbox[2] / anchors[best_anchor][0]),
                    np.log(bbox[3] / anchors[best_anchor][1]),
                    bbox_classes
                ], dtype=np.float32)
            matching_true_boxes[y_cell, x_cell, best_anchor] = adj_bbox
    return detectors_mask, matching_true_boxes


class HandLocalizationData:
    def __init__(self, images_dir_path, labels_file_path, resized_res=(416, 416), max_boxes=None):
        self.images_dir_path = images_dir_path
        self.labels_file_path = labels_file_path
        self.num_data = len(os.listdir(images_dir_path))
        self.all_bboxes = self._load_train_labels()
        print("Number of training images:", self.num_data)
        if max_boxes is None:
            self.max_boxes = self._getmax_boxes()
        else:
            self.max_boxes = max_boxes
        self.resized_res = resized_res
        self.anchors = get_anchors()

    def _getmax_boxes(self):
        max_boxes = 0
        for img_bboxes in self.all_bboxes.values():
            if max_boxes < img_bboxes.shape[0]:
                max_boxes = img_bboxes.shape[0]
        return max_boxes

    def _load_train_labels(self):
        return np.load(self.labels_file_path, allow_pickle=True)

    def generator(self, batch_size, num_epochs):
        num_train_steps = self.num_data // batch_size
        img_paths = os.listdir(self.images_dir_path)
        for epoch in range(num_epochs):
            randomized_steps = random.sample(range(num_train_steps), num_train_steps)
            for batch in randomized_steps:
                batch_img_paths = img_paths[batch * batch_size:(batch + 1) * batch_size]
                batch_images = np.empty((batch_size, self.resized_res[0], self.resized_res[1], 3))
                batch_bboxes = np.empty((batch_size, self.max_boxes, 5))
                detecotrs_mask_1 = np.empty((batch_size, 26, 26, 5, 1))
                detecotrs_mask_2 = np.empty((batch_size, 13, 13, 5, 1))
                matching_true_boxes_1 = np.empty((batch_size, 26, 26, 5, 5))
                matching_true_boxes_2 = np.empty((batch_size, 13, 13, 5, 5))
                for i, img_path in enumerate(batch_img_paths):
                    full_img_path = os.path.join(self.images_dir_path, img_path)
                    img = cv2.imread(full_img_path)
                    img_height, img_width = img.shape[:2]
                    batch_images[i] = preprocess_img(img, self.resized_res, normalize=True)
                    img_bboxes = self.all_bboxes.get(img_path)
                    if img_bboxes.shape[0] < self.max_boxes:
                        zero_pad = np.zeros((self.max_boxes - img_bboxes.shape[0], 5), dtype=np.float32)
                        img_bboxes = np.vstack((img_bboxes, zero_pad))
                    batch_bboxes[i] = normalize_bboxes(img_bboxes, img_width, img_height)
                    detecotrs_mask_1[i], matching_true_boxes_1[i] = preprocess_bboxes(batch_bboxes[i], self.anchors[0], self.resized_res, num_pool_layers=4)
                    detecotrs_mask_2[i], matching_true_boxes_2[i] = preprocess_bboxes(batch_bboxes[i], self.anchors[1], self.resized_res, num_pool_layers=5)
                yield [batch_images, batch_bboxes, detecotrs_mask_1, detecotrs_mask_2, matching_true_boxes_1, matching_true_boxes_2], np.zeros(batch_size)
