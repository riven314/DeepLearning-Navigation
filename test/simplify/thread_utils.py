"""
do heavy lifting job on threading for pyqt5 interface

REFERENCE:
1. now speed ~0.14-0.15s, cpu: 100% in interface

DISCOVERY:
1. [interface] simplify case, 100% CPU in interface, 0.13-0.14s per frame
2. [isolate model run] 60% CPU, 0.1s per frame
3. can set resize image = (427, 240) to speed up
4. so fast when (640, 360) with ensemble size = 1 (< 0.05s in isolation, < 0.07s in interface)

"""
import os
import sys
import time

import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QRunnable
import torch
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
        self.seg_out = None
        

class FrameThread(QThread):
    frame_signal = pyqtSignal(FrameStore)

    def __init__(self, parent = None):
        super().__init__()
        self.load_img()
        self.IS_SILENT = True
        self.model_config = ModelMetaConfig()
        self.frame_store = FrameStore()
    
    def load_img(self):
        self.PATH = os.path.join(os.getcwd(), 'test_cases', 'test.jpg')
        rgb_img = cv2.imread(self.PATH)
        self.rgb_img = rgb_img[:, :, ::-1]

    def run(self):
        """
        continuously feed image into segmentation model repeatedly (to simulate real time workload)

        BREAKDOWN (THREAD MODE):
            [model run] 130 - 145 ms
            [unexplained] ~1-5 ms
        """
        end_loop = time.time()
        while True:
            crt_t = time.time()
            print('time = {:10.4f} s'.format(time.time()))
            print('unexplain = {:10.4f} s'.format(crt_t - end_loop))
            torch.cuda.synchronize()
            start = time.time()
            seg_out = self.model_config.raw_predict(self.rgb_img, self.IS_SILENT)
            seg_out = self.model_config.process_predict(seg_out, self.IS_SILENT)
            torch.cuda.synchronize()
            end = time.time()
            print('model run = {:10.4f} s'.format(end - start))
            start = time.time()
            # store key data at a snapsho
            self.frame_store.rgb_img = self.rgb_img
            self.frame_store.rgb2_img = self.rgb_img
            self.frame_store.rgb3_img = self.rgb_img
            self.frame_store.rgb4_img = self.rgb_img
            self.frame_store.seg_out = seg_out
            self.frame_signal.emit(self.frame_store)
            end_loop = time.time()
            print('emit = {:10.4f} s'.format(end_loop - start))
