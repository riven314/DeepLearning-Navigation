import os
import sys
import time
ROOT_PATH = os.path.join(os.getcwd(), '..', '..')
MOBILENET_PATH = os.path.join(ROOT_PATH, 'mobilenet_segment')
sys.path.append(ROOT_PATH)
sys.path.append(MOBILENET_PATH)

import numpy as np
import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
# stylise pyqt5 interface
import qtmodern.styles
import qtmodern.windows

from thread_utils import FrameStore, FrameThread
from object_utils import FrameObject
from pyqt_utils import convert_qimg
from profiler import profile


class Window(QWidget):
    def __init__(self):
        super().__init__()
        # set up window
        self.title = 'Ready to Rock'
        self.top = 100
        self.left = 100
        self.width = 1280
        self.height = 1280
        self.init_window()
        self.init_thread()
        #self.init_worker()

    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.init_rgbd_layout()
        self.show()

    def init_worker(self):
        self.thread = QThread()
        self.worker = FrameObject()
        self.worker.moveToThread(self.thread)
        self.worker.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'rgb'))
        self.worker.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'seg'))
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def init_thread(self):
        self.f_thread = FrameThread()
        # propagate segmentation color-name mapping to this window
        self.seg_names = self.f_thread.model_config.names
        self.seg_colors = self.f_thread.model_config.colors
        # emit picture to 3 widgets (pixmap)
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'rgb'))
        self.f_thread.frame_signal.connect(lambda frame_store: self.display_pixmap(frame_store, 'seg'))
        self.f_thread.start()

    def init_rgbd_layout(self):
        vbox_layout = QVBoxLayout()
        # set up label
        WIDGET_WIDTH = 484 # 484
        WIDGET_HEIGHT = 240 # 240
        rgb_label = QLabel(self)
        rgb_label.resize(WIDGET_WIDTH, WIDGET_HEIGHT)
        seg_label = QLabel(self)
        seg_label.resize(WIDGET_WIDTH, WIDGET_HEIGHT)
        # assign labels as attribute
        self.rgb_label = rgb_label
        self.seg_label = seg_label
        # stack widgets
        vbox_layout.addWidget(self.rgb_label)
        vbox_layout.addWidget(self.seg_label)
        # logging
        self.setLayout(vbox_layout)

    def display_pixmap(self, frame_store, img_type):
        """
        input:
            frame_store -- FrameStore instance
            img_type -- str, 'rgb', 'depth' or 'seg'
        """
        assert img_type in ['rgb', 'seg'], 'WRONG ARGUMENT img_type'
        if img_type == 'rgb':
            qimg = convert_qimg(frame_store.rgb_img)
            self.rgb_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        else:
            self.seg_qimg = convert_qimg(frame_store.seg_out)
            self.seg_label.setPixmap(QtGui.QPixmap.fromImage(self.seg_qimg))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    qtmodern.styles.dark(app)
    win_modern = qtmodern.windows.ModernWindow(win)
    win_modern.show()
    sys.exit(app.exec_())

