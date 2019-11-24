"""
speed up colorizer
"""
import os
import sys
import time
import timeit
PATH = os.path.join(os.getcwd(), '..', 'd435_camera')
sys.path.append(PATH)

import cv2
import numpy as np
import pyrealsense2 as rs

from camera_config import RGBDhandler

colorizer = rs.colorizer()
RESOLUTION = (1280, 720)
RGB_FORMAT = 'bgr8'
DEPTH_FORMAT = 'z16'
FPS = 30
rs_camera = RGBDhandler(RESOLUTION, RGB_FORMAT, RESOLUTION, DEPTH_FORMAT, FPS)
frames = rs_camera.pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

def rs_colorize(depth_frame, colorizer):
    return np.asanyarray(colorizer.colorize(depth_frame).get_data())

if __name__ == '__main__':
    for i in range(10):
        start = time.time()
        rs_colorize(depth_frame, colorizer)
        end = time.time()
        print('runtime: {} s'.format(end - start))