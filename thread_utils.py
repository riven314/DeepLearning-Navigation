"""
do heavy lifting job on threading for pyqt5 interface

ISSUE:
1. received frame is RGB or BGR??
"""
import os
import sys
import time
import logging

import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import pyrealsense2 as rs

from d435_module.camera_config import RGBDhandler
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
        self.depth_img = None
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
        # passing to interface
        self.frame_store = FrameStore()
    
    def run(self):
        #ptvsd.debug_this_thread()
        while True:
            frames = self.rs_camera.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            rgb_frame = frames.get_color_frame() # uint 8
            depth_frame = frames.get_depth_frame() # unit 8
            color_image = np.asanyarray(rgb_frame.get_data())
            #depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            seg_out = self.model_config.raw_predict(color_image)
            seg_out = self.model_config.process_predict(seg_out)
            #display_image = np.concatenate((color_image, depth_colormap), axis=1)
            # store key data at a snapshot
            self.frame_store.rgb_img = color_image
            self.frame_store.depth_img = depth_colormap
            self.frame_store.seg_out = seg_out
            self.frame_signal.emit(self.frame_store)