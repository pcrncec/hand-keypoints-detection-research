import tensorflow as tf
from hand_detection.utils import decode_conv_output


class YoloLoss(tf.keras.layers.Layer):
    def __init__(self, anchors, num_classes):
        super(YoloLoss, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes

    def call(self, inputs):
        conv_output_1 = inputs[0][0]
        conv_output_2 = inputs[0][1]
        true_boxes = inputs[1]
        detectors_mask_1 = inputs[2]
        detectors_mask_2 = inputs[3]
        matching_true_boxes_1 = inputs[4]
        matching_true_boxes_2 = inputs[5]
        coord_loss_1, conf_loss_1, class_loss_1 = self._calculate_loss(conv_output_1, true_boxes, detectors_mask_1,
                                                                       matching_true_boxes_1, self.anchors[0], 5)
        coord_loss_2, conf_loss_2, class_loss_2 = self._calculate_loss(conv_output_2, true_boxes, detectors_mask_2,
                                                                       matching_true_boxes_2, self.anchors[1], 5)

        total_coord_loss = coord_loss_1 + coord_loss_2
        total_conf_loss = conf_loss_1 + conf_loss_2
        total_class_loss = class_loss_1 + class_loss_2
        total_loss = total_coord_loss + total_conf_loss + total_class_loss
        # tf.print(f" loss: {total_loss}, box_coord_loss: {total_coord_loss}, conf_loss: {total_conf_loss}, class_loss: {total_class_loss}")
        return total_loss

    def _calculate_loss(self, conv_output, true_boxes, detectors_mask, matching_true_boxes, anchors, object_scale,
                        iou_thresh=0.6):
        pred_xy, pred_wh, pred_confidence, pred_class_prob = decode_conv_output(self.num_classes, conv_output, anchors)
        conv_shape = tf.shape(conv_output)
        true_boxes_shape = tf.shape(true_boxes)
        conv_dim = conv_shape[1]
        num_anchors = len(anchors)

        reshaped_conv_output = tf.reshape(conv_output, [-1, conv_dim, conv_dim, num_anchors, self.num_classes + 5])
        pred_boxes = tf.concat((tf.sigmoid(reshaped_conv_output[..., :2]), reshaped_conv_output[..., 2:4]), axis=-1)

        pred_xy = tf.expand_dims(pred_xy, 4)
        pred_wh = tf.expand_dims(pred_wh, 4)
        pred_wh_half = pred_wh / 2.
        pred_xy_mins = pred_xy - pred_wh_half
        pred_xy_maxes = pred_xy + pred_wh_half

        true_boxes = tf.reshape(true_boxes, [true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]])
        true_xy = true_boxes[..., :2]
        true_wh = true_boxes[..., 2:4]
        true_wh_half = true_wh / 2.
        true_xy_mins = true_xy - true_wh_half
        true_xy_maxes = true_xy + true_wh_half

        intersect_mins = tf.maximum(pred_xy_mins, true_xy_mins)
        intersect_maxes = tf.minimum(pred_xy_maxes, true_xy_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)

        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        union_areas = pred_areas + true_areas - intersect_areas

        iou_scores = intersect_areas / union_areas
        best_iou_scores = tf.reduce_max(iou_scores, axis=4)
        best_iou_scores = tf.expand_dims(best_iou_scores, axis=-1)

        found_detections = tf.cast(best_iou_scores > iou_thresh, tf.float32)
        no_object_weights = (1 - found_detections) * (1 - detectors_mask)
        no_objects_loss = no_object_weights * tf.square(-pred_confidence)
        objects_loss = (object_scale * detectors_mask * tf.square(1 - pred_confidence))
        confidence_loss = objects_loss + no_objects_loss

        matching_boxes = matching_true_boxes[..., 0:4]
        coordinates_loss = detectors_mask * tf.square(matching_boxes - pred_boxes)

        matching_classes = tf.cast(matching_true_boxes[..., 4], 'int32')
        matching_classes = tf.one_hot(matching_classes, self.num_classes)
        classification_loss = (detectors_mask * tf.square(matching_classes - pred_class_prob))

        coordinates_loss_mean = tf.reduce_mean(tf.reduce_sum(coordinates_loss, axis=[1, 2, 3, 4]))
        confidence_loss_mean = tf.reduce_mean(tf.reduce_sum(confidence_loss, axis=[1, 2, 3, 4]))
        classification_loss_mean = tf.reduce_mean(tf.reduce_sum(classification_loss, axis=[1, 2, 3, 4]))
        return coordinates_loss_mean, confidence_loss_mean, classification_loss_mean
