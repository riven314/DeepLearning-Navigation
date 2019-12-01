"""
simplified version of thread_utils.py for testing
include:
1. alternate static image
2. segmentation module
"""
import os
import sys
import time
ROOT_PATH = os.path.join(os.getcwd(), '..')
MOBILENET_PATH = os.path.join(ROOT_PATH, 'mobilenet_segment')
sys.path.append(ROOT_PATH)
sys.path.append(MOBILENET_PATH)

import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QRunnable
import torch
from model_utils import ModelMetaConfig


class FrameStore:
    def __init__(self):
        self.rgb_img = None
        self.d1_img = None
        self.pred_rgb = None
        self.pred_idx = None
        # time each component
        self.camera_time = None
        self.model_time = None
        self.fps_time = None
        

class FrameThread(QThread):
    frame_signal = pyqtSignal(FrameStore)

    def __init__(self, parent = None):
        super().__init__()
        self.load_img()
        self.IS_SILENT = True
        self.model_config = ModelMetaConfig(root = MOBILENET_PATH)
        self.model_config.ENSEMBLE_N = 1
        self.frame_store = FrameStore()
    
    def load_img(self):
        IMG_PATH = os.path.join(os.getcwd(), '..', 'test_cases', 'test_obj_avoid_rgb.jpg')
        D1_PATH = os.path.join(os.getcwd(), '..', 'test_cases', 'test_obj_avoid_d1.png')
        rgb_img = cv2.imread(IMG_PATH)
        d1_img = cv2.imread(D1_PATH, cv2.IMREAD_GRAYSCALE)
        # slow without img.copy()
        self.rgb_img = rgb_img[:, :, ::-1].copy()
        self.d1_img = d1_img

    def run(self):
        """
        continuously feed image into segmentation model repeatedly (to simulate real time workload)

        BREAKDOWN (THREAD MODE):
            [model run] 130 - 145 ms
            [unexplained] ~1-5 ms
        """
        crt_t = time.time()
        while True:
            self.frame_store.fps_time = 0 if time.time() - crt_t < 0.01 else 1 / (time.time() - crt_t)
            crt_t = time.time()
            torch.cuda.synchronize()
            model_start = time.time()
            rgb_img = self.rgb_img
            seg_out = self.model_config.raw_predict(rgb_img, self.IS_SILENT)
            pred_idx, pred_rgb = self.model_config.process_predict(seg_out, self.IS_SILENT)
            torch.cuda.synchronize()
            model_end = time.time()
            self.frame_store.model_time = model_end - model_start
            # store key data at a snapsho
            self.frame_store.rgb_img = rgb_img
            self.frame_store.pred_rgb = pred_rgb
            self.frame_store.pred_idx = pred_idx
            # resize d1 depth image
            self.frame_store.d1_img = cv2.resize(self.d1_img, self.model_config.RESIZE, interpolation = cv2.INTER_LINEAR)
            self.frame_signal.emit(self.frame_store)