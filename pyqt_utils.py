"""
do heavy lifting jobs for pyqt5 interface setting
"""
import cv2
from PyQt5 import QtGui
from PyQt5.QtCore import Qt


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