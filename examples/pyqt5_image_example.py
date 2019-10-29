"""
REFERENCE:
1. QImage, pyqt5 and opencv https://zhuanlan.zhihu.com/p/31810054

QUESTIONS:
1. Why need to do RGB2BGR?
"""
import os
import sys

import cv2
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5.QtCore import Qt, QRect

class myLabel(QLabel):
    x0, y0, x1, y1 = 0, 0, 0, 0
    flag = False
    SAVE_PATH = os.path.join('images', 'save.png')

    # override default mousePressEvent
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    def mouseReleaseEvent(self, event):
        self.flag = False

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            # why .update()?
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        # DRAW BOUNDING BOX 
        rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        # why pass self to QPainter?
        painter = QPainter(self)
        # define the bounding lines (e.g. thickness, color, locations)
        painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        painter.drawRect(rect)

        # SNAPSHOT TO THE DISK
        pqscreen = QGuiApplication.primaryScreen()
        pixmap2 = pqscreen.grabWindow(self.winId(), self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        pixmap2.save(self.SAVE_PATH)

class Window(QWidget):
    IMAGE_PATH = os.path.join('images', 'pytorch.png')

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.resize(2400, 1600)
        self.setWindowTitle('Demo Image')
        self.lb = myLabel(self)
        # if too small can't display the whole image
        self.lb.setGeometry(QRect(140, 30, 2200, 1200))

        
        img = cv2.imread(self.IMAGE_PATH)
        h, w, _ = img.shape
        bytesPerLine = 3 * w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        qimg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.lb.setPixmap(pixmap)
        self.lb.setCursor(Qt.CrossCursor)

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())