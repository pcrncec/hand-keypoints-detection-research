from models.tiny_yolo import TinyYolo
from models.yolo_loss import YoloLoss
import tensorflow as tf
from tensorflow.keras.layers import Input


def create_model(anchors, max_boxes, num_classes, show_full_summary=True):
    img_input = Input((416, 416, 3))
    bbox_input = Input((max_boxes, 5))
    detect_mask_input_1 = Input((26, 26, 5, 1))
    detect_mask_input_2 = Input((13, 13, 5, 1))
    matching_boxes_input_1 = Input((26, 26, 5, 5))
    matching_boxes_input_2 = Input((13, 13, 5, 5))

    tiny_yolo_body = TinyYolo(anchors, num_classes=4)
    body_model = tf.keras.Model(img_input, tiny_yolo_body(img_input))
    loss_layer = YoloLoss(anchors, num_classes=num_classes)(
        [body_model.output, bbox_input, detect_mask_input_1, detect_mask_input_2,
         matching_boxes_input_1, matching_boxes_input_2])

    model = tf.keras.Model([body_model.inputs, bbox_input, detect_mask_input_1, detect_mask_input_2,
                            matching_boxes_input_1, matching_boxes_input_2], loss_layer)
    if show_full_summary:
        full_model_summary(model)
    return model


def full_model_summary(model):
    for submodel in model.layers:
        if hasattr(submodel, 'summary'):
            submodel.summary()
    print("==================================================================================================")
    print("==================================================================================================")
    model.summary()


def get_prediction_model(anchors, max_boxes, num_classes, weights_path):
    model_with_loss = create_model(anchors, max_boxes, num_classes, show_full_summary=False)
    model_with_loss.load_weights(weights_path)
    prediction_model = tf.keras.Model(model_with_loss.layers[0].input, model_with_loss.layers[1].output)
    return prediction_model
