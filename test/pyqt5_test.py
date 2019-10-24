"""
test if pyqt5 interface can do the following:
1. stream the same video in 2 separate windows 
2. retrieve pixel information depending on your cursor location

REFERENCE
1. pyqt5 + video stream in opencv https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv?fbclid=IwAR1oUhFLXk2HVkrs4yTqV2yvtE933A8fSjEBb853LK7HM3cIHrVk-ElurUg
2. get pixel information in pyqt5 https://stackoverflow.com/questions/3504522/pyqt-get-pixel-position-and-value-when-mouse-click-on-the-image
3. QThread explanation https://medium.com/@webmamoffice/getting-started-gui-s-with-python-pyqt-qthread-class-1b796203c18c
4. PyQt5 tutorial http://zetcode.com/gui/pyqt5/

LOG
[22/10/2019]
- What is QThread?
- How does the image pop out in GUI?

[24/10/2019]
- What is QImage? QThread? QPixmap?
"""
import os
import sys

import numpy as np
import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

class Thread(QThread):
    # a signal called "changePixmap" that take an argument QImage
    # QImage is an image representation that allows direct access to pixel data
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        th = Thread(self)
        # signal is connected to a slot using .connect()
        th.changePixmap.connect(self.setImage)
        th.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())