"""
PROCEDURES:
1. fix the speed issue
2. refactor the code
3. repull the github repo

QUESTIONS:
1. how to unprint required logging level below
2. consolidate all configurations
"""
import os
import sys
import time
import logging

# suppress unknown warning (e.g. QWindowsContext: OleInitialize() failed)
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

from d435_module.camera_config import RGBDhandler
import pyrealsense2 as rs

# config for logging
LOG_FILE = os.path.join('logs', 'log.txt')
LOG_FORMAT = '%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s'
formatter = logging.Formatter(LOG_FORMAT)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

# config for realsense camera
RESOLUTION = (1280, 720)
RGB_FORMAT = 'bgr8'
DEPTH_FORMAT = 'z16'
FPS = 30

########################### FOR MOBILENET ###########################
import os
import sys
import time
seg_module_path = os.path.join(os.getcwd(), 'mobilenet_segment')
sys.path.append(seg_module_path)

import numpy as np
from scipy.io import loadmat
import csv
from torchvision import transforms

from webcam_test import IamgeLoad, setup_model, predict, process_predict
from config.defaults import _C as cfg

#Define the color dict
COLOR_PLATE_PATH = os.path.join(os.getcwd(), 'mobilenet_segment', 'data', 'color150.mat')
PRINT_PATH = os.path.join(os.getcwd(), 'mobilenet_segment', 'data', 'object150_info.csv')
DATA_PATH = os.path.join(os.getcwd(), 'mobilenet_segment', 'test_set', 'cls1_rgb.npy')
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


img = np.load(DATA_PATH)
cfg_path = os.path.join(os.getcwd(), 'mobilenet_segment', 'config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
#cfg_path="config/ade20k-resnet18dilated-ppm_deepsup.yaml"

model = setup_model(cfg_path, ROOT, gpu=0)
print('model setup complete')
########################### FOR MOBILENET ###########################


def convert_qimg(frame, win_width = 420, win_height = 240):
    """
    convert from cv2 frame to QImage frame.

    input:
        frame -- np array, in BGR channel
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    byte_per_line = c * w
    qimg = QtGui.QImage(img.data, w, h, byte_per_line, QtGui.QImage.Format_RGB888)\
            .scaled(win_width, win_height, Qt.KeepAspectRatio)
    return qimg


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
        super().__init__()
        self.rs_camera = RGBDhandler(RESOLUTION, RGB_FORMAT, RESOLUTION, DEPTH_FORMAT, FPS)
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        self.frame_store = FrameStore()
        self.mobilenet = model
    
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
            seg_out = self.predict_seg(color_image)
            #display_image = np.concatenate((color_image, depth_colormap), axis=1)
            # store key data at a snapshot
            self.frame_store.rgb_img = color_image
            self.frame_store.depth_img = depth_colormap
            self.frame_store.seg_out = seg_out
            self.frame_signal.emit(self.frame_store)

    def predict_seg(self, rgb_img):
        Iamge = IamgeLoad(rgb_img, WIDTH, HEIGHT)
        predictions = predict(model, Iamge, RESIZE_NUM ,gpu=0)
        _, pred_color = process_predict(predictions, colors, names)
        #qimg = convert_qimg(pred_color)
        #self.seg_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        return pred_color


class Window(QWidget):
    def __init__(self):
        super().__init__()
        # setup logger
        self.logger = logging.getLogger('Window')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        # set up window
        self.title = 'Ready to Rock'
        self.top = 100
        self.left = 100
        self.width = 1280
        self.height = 1280
        self.init_window()
        self.logger.info('window setup complete')

    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.init_rgbd_layout()
        self.init_thread()
        self.show()

    def init_thread(self):
        self.f_thread = FrameThread()
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'rgb'))
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'depth'))
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'seg'))
        self.logger.info('framing thread setup complete')
        self.f_thread.start()

    def init_rgbd_layout(self):
        vbox_layout = QVBoxLayout()
        # set up label
        rgb_label = QLabel(self)
        rgb_label.resize(420, 240)
        rgb_title = QLabel('RGB Image')
        depth_label = QLabel(self)
        depth_label.resize(420, 240)
        depth_title = QLabel('Depth Image')
        seg_label = QLabel(self)
        seg_label.resize(420, 240)
        seg_title = QLabel('Segmentation Output')
        # assign labels as attribute
        self.rgb_label = rgb_label
        self.depth_label = depth_label
        self.seg_label = seg_label
        # stack widgets
        vbox_layout.addWidget(rgb_title)
        vbox_layout.addWidget(self.rgb_label)
        vbox_layout.addWidget(depth_title)
        vbox_layout.addWidget(self.depth_label)
        vbox_layout.addWidget(seg_title)
        vbox_layout.addWidget(self.seg_label)
        # logging
        self.logger.info('widget setup complete')
        self.setLayout(vbox_layout)

    def display_pixmap(self, frame_store, img_type):
        """
        input:
            frame_store -- FrameStore instance
            img_type -- str, 'rgb', 'depth' or 'seg'
        """
        assert img_type in ['rgb', 'depth', 'seg'], 'WRONG ARGUMENT img_type'
        if img_type == 'rgb':
            qimg = convert_qimg(frame_store.rgb_img)
            self.rgb_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        elif img_type == 'depth':
            qimg = convert_qimg(frame_store.depth_img)
            self.depth_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        else:
            qimg = convert_qimg(frame_store.seg_out)
            self.seg_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def run_mobilenet(self, frame_store):
        img = frame_store.rgb_img
        Iamge = IamgeLoad(img, WIDTH, HEIGHT)
        predictions = predict(model, Iamge, RESIZE_NUM ,gpu=0)
        seg, pred_color = process_predict(predictions, colors, names)
        qimg = convert_qimg(pred_color)
        self.seg_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Window()
    form.show()
    sys.exit(app.exec_())

