"""
do heavy lifting job on threading for pyqt5 interface

ISSUE:
1. received frame is RGB or BGR??
2. how to profile multithreading application?
"""
import os
import sys
import time
import logging

import numpy as np
import cv2
import torch
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import pyrealsense2 as rs

from d435_camera.camera_config import RGBDhandler
from model_utils import ModelMetaConfig

class FrameStore:
    """
    store key data at a snapshot
    (i.e. rgb image, depth image and segmentation result)

    attribute:
        rgb_img -- np array
        depth_img -- np array
        seg_out -- np array 
    
    * all have same dimension
    """
    def __init__(self):
        self.rgb_img = None
        self.depth_1c_img = None
        self.depth_3c_img = None
        self.seg_out = None
        

class FrameThread(QThread):
    frame_signal = pyqtSignal(FrameStore)

    def __init__(self, parent = None):
        """
        carry 3 instances: 
            1. realsense class instance
            2. model metaconfig instance
            3. frame store instance
        """
        super().__init__()
        # realsense related
        self.RESOLUTION = (1280, 720)
        self.RGB_FORMAT = 'bgr8'
        self.DEPTH_FORMAT = 'z16'
        self.FPS = 30
        self.rs_camera = RGBDhandler(self.RESOLUTION, self.RGB_FORMAT, 
                                     self.RESOLUTION, self.DEPTH_FORMAT, 
                                     self.FPS)
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        # model related
        self.model_config = ModelMetaConfig()
        self.IS_SILENT = True
        # passing to interface
        self.frame_store = FrameStore()
    
    def run(self):
        """
        RUNTIME BREAKDOWN:
            [realsense]
                wait frame: ~0-1 ms
                get rgbd frame: ~0-1 ms
                align: ~25 ms
                frame to np: 0 ms
                colorize depth: 16 ms
            [model]
                process + predict + visualize: 120-130 ms
            [tidy up]
                total: 1 ms
            [unexplain after loop]
                total: 5 ms

        REFERENCE:
        1. [align] low FPS when align is used (align is CPU intensive): https://github.com/IntelRealSense/librealsense/issues/5218
        2. [align] how to enable cuda in realsense: https://github.com/IntelRealSense/librealsense/issues/4905#issuecomment-533854888
        3. [align] align runtime on CUDA v.s. CPU: https://github.com/IntelRealSense/librealsense/pull/2670
        4. [align] another discussion on low FPS for align: https://github.com/IntelRealSense/librealsense/issues/2321
        5. [pyqt] proper way to construct QThread: https://forum.qt.io/topic/85826/picamera-significant-delay-and-low-fps-but-low-cpu-and-memory-usage/2
        """
        end_loop = time.time()
        while True:
            torch.cuda.synchronize()
            t = time.time()
            print('time: {:10.4f} s'.format(t))
            print('unexplained: {:10.4f} s'.format(t - end_loop))
            start = time.time()
            frames = self.rs_camera.pipeline.wait_for_frames()
            end = time.time()
            print('realsense wait: {:10.4f} s'.format(end - start))
            start = time.time()
            frames = self.align.process(frames)
            end = time.time()
            print('realsense align: {:10.4f} s'.format(end - start))
            start = time.time()
            rgb_frame = frames.get_color_frame() # uint 8
            depth_frame = frames.get_depth_frame() # unit 8
            end = time.time()
            print('realsense get frame: {:10.4f} s'.format(end - start))
            start = time.time()
            color_image = np.asanyarray(rgb_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            end = time.time()
            print('realsense frame np: {:10.4f} s'.format(end - start))
            start = time.time()
            depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            end = time.time()
            print('realsense colorize: {:10.4f} s'.format(end - start))
            torch.cuda.synchronize()
            start = time.time()
            seg_out = self.model_config.raw_predict(color_image, is_silent = self.IS_SILENT)
            seg_out = self.model_config.process_predict(seg_out, is_silent = self.IS_SILENT)
            torch.cuda.synchronize()
            end = time.time()
            print('process+predict: {:10.4f} s'.format(end - start))
            start = time.time()
            # store key data at a snapshot
            self.frame_store.rgb_img = color_image
            self.frame_store.depth_1c_img = depth_image
            self.frame_store.depth_3c_img = depth_colormap
            self.frame_store.seg_out = seg_out
            self.frame_signal.emit(self.frame_store)
            end_loop = time.time()
            print('end+emit: {:10.4f} s'.format(end_loop - start))
            
