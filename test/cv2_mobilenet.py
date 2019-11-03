"""
apply mobilenet on realsense camera and print it out cv2
"""
##################### SETUP MOBILENET #####################
import os
import sys
import time
root_path = os.path.join(os.getcwd())
seg_module_path = os.path.join(os.getcwd(),'mobilenet_segment')
sys.path.append(root_path)
sys.path.append(seg_module_path)

import numpy as np
from scipy.io import loadmat
import csv
from torchvision import transforms

from webcam_test import IamgeLoad, setup_model, predict, process_predict
from config.defaults import _C as cfg

#Define the color dict
COLOR_PLATE_PATH = os.path.join('mobilenet_segment', 'data', 'color150.mat')
PRINT_PATH = os.path.join('mobilenet_segment', 'data', 'object150_info.csv')
DATA_PATH = os.path.join('mobilenet_segment', 'test_set', 'cls1_rgb.npy')
ROOT = os.path.join(os.getcwd(), 'mobilenet_segment')
WIDTH = 424
HEIGHT = 240
RESIZE_NUM = 3

colors = loadmat(COLOR_PLATE_PATH)['colors']
names = {}
with open(PRINT_PATH) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0] 

cfg_path = os.path.join('mobilenet_segment', 'config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
model = setup_model(cfg_path, ROOT, gpu=0)

##################### REALSENSE WITH CV2 #####################
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        Iamge = IamgeLoad(color_image, WIDTH, HEIGHT)
        pred = predict(model, Iamge, RESIZE_NUM, gpu = 0)
        seg, pred_color = process_predict(pred, colors, names)
        # Stack both images horizontally
        #images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('Prediction', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Prediction', pred_color)
        cv2.imshow('RGB', color_image)

        cv2.waitKey(1)
finally:
    # Stop streaming
    pipeline.stop()

#predictions = predict(model, Iamge, RESIZE_NUM ,gpu=0)