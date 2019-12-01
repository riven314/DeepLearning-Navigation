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
    def __init__(self):
        # store camera output
        self.rgb_img = None
        self.d1_img = None
        # store segmentation result
        self.pred_rgb = None
        self.pred_idx = None
        # time each component
        self.camera_time = None
        self.model_time = None
        self.fps_time = None
        

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
        #self.colorizer = rs.colorizer()
        # model related
        self.model_config = ModelMetaConfig()
        self.model_config.ENSEMBLE_N = 1
        self.IS_SILENT = True
        # passing to interface
        self.frame_store = FrameStore()
    
    def run(self):
        crt_t = time.time()
        while True:
            # time FPS
            self.frame_store.fps_time = 0 if time.time() - crt_t < 0.01 else 1 / (time.time() - crt_t)
            crt_t = time.time()
            # camera module
            torch.cuda.synchronize()
            camera_start = time.time()
            frames = self.rs_camera.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            rgb_frame = frames.get_color_frame() 
            depth_frame = frames.get_depth_frame() 
            color_image = np.asanyarray(rgb_frame.get_data()) # uint 8
            depth_image = np.asanyarray(depth_frame.get_data()) # unit 16
            torch.cuda.synchronize()
            camera_end = time.time()
            self.frame_store.camera_time = camera_end - camera_start
            # segmentation module
            model_start = time.time()
            seg_out = self.model_config.raw_predict(color_image, is_silent = self.IS_SILENT)
            pred_idx, pred_rgb = self.model_config.process_predict(seg_out, is_silent = self.IS_SILENT)
            torch.cuda.synchronize()
            model_end = time.time()
            self.frame_store.model_time = model_end - model_start
            # store items in frame_store
            self.frame_store.rgb_img = color_image
            self.frame_store.pred_idx = pred_idx
            self.frame_store.pred_rgb = pred_rgb
            self.frame_store.d1_img = cv2.resize(depth_image, 
                                                 self.model_config.RESIZE, 
                                                 interpolation = cv2.INTER_LINEAR)
            self.frame_signal.emit(self.frame_store)
            
