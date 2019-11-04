"""
PROCEDURES:
1. fix the speed issue
2. refactor the code
3. repull the github repo

QUESTIONS:
1. how to unprint required logging level below
2. consolidate all configurations
3. very CPU intensive (100%), why?
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
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

from d435_camera.camera_config import RGBDhandler
from thread_utils import FrameStore, FrameThread
from pyqt_utils import convert_qimg
from profiler import profile

# config for logging
LOG_FILE = os.path.join('logs', 'log.txt')
LOG_FORMAT = '%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s'
formatter = logging.Formatter(LOG_FORMAT)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)


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
        # propagate segmentation color-name mapping to this window
        self.seg_names = self.f_thread.model_config.names
        self.seg_colors = self.f_thread.model_config.colors
        # emit picture to 3 widgets (pixmap)
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'rgb'))
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'depth'))
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'seg'))
        self.logger.info('framing thread setup complete')
        self.f_thread.start()

    def init_rgbd_layout(self):
        vbox_layout = QVBoxLayout()
        # set up label
        rgb_label = QLabel(self)
        rgb_label.resize(484, 240)
        rgb_title = QLabel('RGB Image')
        depth_label = QLabel(self)
        depth_label.resize(484, 240)
        depth_label.setObjectName('depth')
        depth_label.mousePressEvent = self.query_depth
        depth_title = QLabel('Depth Image')
        seg_label = QLabel(self)
        seg_label.resize(484, 240)
        # set mouse interactive on segmentation widget
        seg_label.setObjectName('seg')
        seg_label.mousePressEvent = self.query_segment
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
            qimg = convert_qimg(frame_store.depth_3c_img)
            # store depth 1-channel map for distance query
            self.depth_1c_map = frame_store.depth_1c_img
            self.depth_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        else:
            self.seg_qimg = convert_qimg(frame_store.seg_out)
            #self.seg_pixmap = QtGui.QPixmap.fromImage(qimg)
            self.seg_label.setPixmap(QtGui.QPixmap.fromImage(self.seg_qimg))

    def query_segment(self, event):
        # x, y coordinate relative to the widget size
        x = event.pos().x()
        y = event.pos().y()
        obj = self.query_segment_mapping(x, y)
        print('x = {}; y = {}, obj = {}'.format(x, y, obj))

    def query_segment_mapping(self, x, y):
        """
        given pixel coordinate, query its predicted object
        """
        c = self.seg_qimg.pixel(x, y)
        #c_qobj = QColor(c)
        c_rgb = QColor(c).getRgb()
        rgb = [c_rgb[2], c_rgb[1], c_rgb[0]]
        idx = np.where((self.seg_colors == rgb).all(axis = 1))[0][0]
        obj = self.seg_names[idx + 1]
        return obj

    def query_depth(self, event):
        # x,y coordinate relative to the widget size
        x = event.pos().x()
        y = event.pos().y()
        distance = self.depth_1c_map[x, y]
        print('x = {}; y = {}; distance = {}'.format(x, y, distance))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Window()
    form.show()
    sys.exit(app.exec_())

