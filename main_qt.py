"""
need to clarify:
1. QThread.run
2. @pyqtSlot(QImage)
3. setImage()

TO DO:
1. get two threads on rgb and depth
2. setup logger for debugging

REFERENCE:
1. [github] QT5 Threads not captured in the debugger via VSCode: https://github.com/microsoft/ptvsd/issues/428
2. [github] when use QThread Debug PyQt app, breakpoint not work,while wing IDE can debug: https://github.com/Microsoft/vscode-python/issues/176
3. [github] debug for QThread by ptvsd: https://github.com/microsoft/ptvsd/issues/1189#issuecomment-468406399
"""
import os
import sys
import time

import numpy as np
import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

from d435_module.camera_config import RGBDhandler
import pyrealsense2 as rs
import ptvsd


def convert_qimg(frame, win_width = 640*2, win_height = 480*2):
    """
    convert from cv2 frame to QImage frame
    """
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb_img.shape
    byte_per_line = c * w
    img = QtGui.QImage(rgb_img.data, w, h, byte_per_line, QtGui.QImage.Format_RGB888)\
               .scaled(win_width, win_height, Qt.KeepAspectRatio)
    return img


class RealsenseThread(QThread):
    # set up signal 
    changePixmap = pyqtSignal(QImage)
    #changePixmap = pyqtSignal(np.array)

    # set up camera
    resolution = (1280, 720)
    rs_camera = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

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
            display_image = np.concatenate((color_image, depth_colormap), axis=1)
            qimg = convert_qimg(display_image)
            self.changePixmap.emit(qimg)
            #self.changePixmap.emit(display_image)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'CAPSTONE DEMO'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    # convert method -> Qt slot (not necessary, but can reduce memory and slightly faster)
    @pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        #ptvsd.debug_this_thread()
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    #@pyqtSlot()
    #def setImage(self, nd_img):
    #    qimg = convert_qimg(nd_img)
    #    self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # set whole window size at start
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        # move x units of right, y units down
        self.label.move(280, -120)
        self.label.resize(640 * 2, 480 * 2)

        # create and run thread
        #ptvsd.debug_this_thread()
        th = RealsenseThread(self)
        # signal is connected to a slot using .connect()
        th.changePixmap.connect(self.setImage)
        th.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())