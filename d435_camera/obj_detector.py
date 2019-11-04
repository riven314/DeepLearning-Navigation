"""
AUTHOR: Alex Lau

SUMMARY
do all heavy-lifting jobs for object detection:
1. load in model weight
2. model inference 
3. bounding box processing
4. distance inference from inference result

REFERENCE
1. object detection + distance to object ipynb
    - https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
2. what does cv2.dnn.blobFromImage is doing (data pre-processing step)
    - https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

LOG
[08/10/2019]
- SSD MobileNet only support square image so far ...
- need to conversion between rectangular image to square image
"""
import os
import sys

import numpy as np
import cv2


def get_class_name(model_type = 'SSD'):
    """
    get label name for different models

    input:
        model_type -- str, 'SSD' ... etc
    """
    if model_type == 'SSD':
        label = ("background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor")
    return label


def load_model(prototxt_path, model_path):
    print('prototxt_path: {}'.format(prototxt_path))
    print('model_path: {}'.format(model_path))
    # fail fast
    assert os.path.isfile(prototxt_path), 'WRONG INPUT prototxt_path'
    assert os.path.isfile(model_path), 'WRONG INPUT model_path'
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print('MODEL LOADED IN')
    return model


def cv2_normalize(img, scale = 0.007843, mean_val = 127.53, height = 300):
    """
    handler for normalizing the image before feeding it into model 
    default is scalar mean and scalar scale for SSD MobileNet, and image input is a square

    input:
        img -- np array, (h, w, c)
        scale -- float/ list, channel-wise std
        mean_val -- float/ list, channel-wise mean
        height -- int, size of model input
    output:
        blob -- np array, (batch, c, h, w), image after pre-processing
    """
    blob = cv2.dnn.blobFromImage(img, scale, (height, height), mean_val, False)
    return blob


def feed_model(model, blob):
    model.setInput(blob, 'data')
    out = model.forward('detection_out')
    return out


def unpack_model_output(out_last):
    """
    input:
        out_last of the i-th prediction is out[0, 0, i, :]
    output:
        label -- float, label number
        conf -- float, confidence of prediction
        xmin -- float, relative position of bounding box
        ymin -- float, relative position of bounding box
        xmax -- float, relative position of bounding box
        ymax -- float, relative position of bounding box
    """
    _, label, conf, xmin, ymin, xmax, ymax = out_last
    return label, conf, xmin, ymin, xmax, ymax


def write_a_box_on_img(img, text, box_tup, expect_height):
    """
    write a single bounding box and text on an image IN-PLACE

    input:
        img -- np array, image to be added bounding box and label text
        text -- str, text added on the image (typically a label name)
        box_tup -- tuple, (xmin, ymin, xmax, ymax)
        expect_height -- int, expect size of image (both width and height)
    output:
        img -- np array, image with a bounding box and label added
    """
    xmin, ymin, xmax, ymax = box_tup
    cv2.rectangle(img, 
                  (int(xmin * expect_height), int(ymin * expect_height)), 
                  (int(xmax * expect_height), int(ymax * expect_height)), 
                  (255, 255, 255),  # color
                  2) # thickness
    cv2.putText(img, text, 
                (int(xmin * expect_height), int(ymin * expect_height) - 5),
                cv2.FONT_HERSHEY_COMPLEX, 
                0.5, # font scale
                (255,255,255)) # color
    return img


def plot_prediction(img, out, class_name, expect_height = 300):
    """
    plot bounding boxes with labels on an image with COPY

    input:
        img -- np array, (expect_height, expect_height, 3), image for model input 
        out -- np array, (., ., # predictions, prediction output)
        expect_height -- int, expect size of img (both width and height)
    """
    # fail fast sanity check
    h, w, _ = img.shape
    assert h == w == expect_height, 'WRONG SIZE IN img AGAINST expect_height'
    # make a copy to avoid contamination
    for i, out_last in enumerate(out[0, 0, :, :]):
        label_id, conf, xmin, ymin, xmax, ymax = unpack_model_output(out_last)
        label_name = class_name[label_id]
        box_tup = (xmin, ymin, xmax, ymax)
         
    

if __name__ == '__main__':
    pass

