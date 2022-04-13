import numpy as np
import os
from random import sample
import cv2


def preprocess_img(img, resized_res, normalize=True):
    if img.shape[:2] != resized_res:
        img = cv2.resize(img, resized_res, interpolation=cv2.INTER_NEAREST)
    img_arr = np.array(img, dtype=np.uint8, copy=False)
    if normalize:
        img_arr = img_arr.astype(np.float) / 255.
    return img_arr


def preprocess_keypoints(keypoints, org_img_res):
    return np.divide(keypoints, org_img_res)


def data_generator(train_images_path, keypoints_path, num_epochs, batch_size, img_res=(224, 224)):
    keypoints = np.load(keypoints_path, allow_pickle=True)
    all_img_paths = os.listdir(train_images_path)
    total_steps = len(all_img_paths) // batch_size
    for epoch in range(num_epochs):
        rand_steps = sample(range(total_steps), total_steps)
        for batch in rand_steps:
            batch_img_paths = all_img_paths[batch * batch_size:(batch + 1) * batch_size]
            batch_images = np.empty((batch_size, img_res[0], img_res[1], 3))
            batch_keypoints = np.empty((batch_size, 21, 2))
            for i, img_path in enumerate(batch_img_paths):
                full_img_path = os.path.join(train_images_path, img_path)
                rgb_img = cv2.imread(full_img_path)
                org_img_res = rgb_img.shape[:2]
                batch_images[i] = preprocess_img(rgb_img, img_res)
                img_keypoints = keypoints.get(img_path)
                batch_keypoints[i] = preprocess_keypoints(img_keypoints, org_img_res)
            yield batch_images, batch_keypoints
