import os
import sys

import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout
import qtmodern.styles
import qtmodern.windows


IMAGE_PATH = os.path.join('images', 'camera.png')

class Window(QWidget):
    def __init__(self):
        super().__init__()
        # set up window
        self.TITLE = 'Ready to Rock'
        self.TOP = 100
        self.LEFT = 100
        self.WIDTH = 1280
        self.HEIGHT = 1280
        # set up sub window (widget)
        self.WIDGET_WIDTH = 484
        self.WIDGET_HEIGHT = 240
        self.init_window()

    def init_window(self):
        self.setWindowTitle(self.TITLE)
        self.setGeometry(self.TOP, self.LEFT, self.WIDTH, self.HEIGHT)
        self.init_top_groupbox()
        self.init_bottom_groupbox()
        #self.init_thread()
        self.init_gridlayout()
        self.show()

    def init_gridlayout(self):
        main_layout = QGridLayout()
        main_layout.addWidget(self.top_groupbox, 1, 0)
        main_layout.addWidget(self.bottom_groupbox, 2, 0)
        self.setLayout(main_layout)
    
    def init_top_groupbox(self):
        self.top_groupbox = QGroupBox('From Realsense Camera')
        self.init_topleft_groupbox()
        self.init_topright_groupbox()
        layout = QHBoxLayout()
        layout.addWidget(self.topleft_groupbox)
        layout.addWidget(self.topright_groupbox)
        self.top_groupbox.setLayout(layout)

    def init_topleft_groupbox(self):
        self.topleft_groupbox = QGroupBox('RGB')
        rgb_label = QLabel(self)
        rgb_label.resize(self.WIDGET_WIDTH / 2, self.WIDGET_HEIGHT)
        # read image
        qimg = self.get_qimage(IMAGE_PATH)
        pixmap = QPixmap.fromImage(qimg)
        rgb_label.setPixmap(pixmap)
        layout = QHBoxLayout()
        layout.addWidget(rgb_label)
        self.topleft_groupbox.setLayout(layout)

    def init_topright_groupbox(self):
        self.topright_groupbox = QGroupBox('Depth')
        # depth colormap
        depth_3c_label = QLabel(self)
        depth_3c_label.resize(self.WIDGET_WIDTH, self.WIDGET_HEIGHT)
        qimg = self.get_qimage(IMAGE_PATH)
        pixmap = QPixmap.fromImage(qimg)
        depth_3c_label.setPixmap(pixmap)
        # depth map
        depth_1c_label = QLabel(self)
        depth_1c_label.resize(self.WIDGET_WIDTH, self.WIDGET_HEIGHT)
        qimg = self.get_qimage(IMAGE_PATH)
        pixmap = QPixmap.fromImage(qimg)
        depth_1c_label.setPixmap(pixmap)
        layout = QHBoxLayout()
        layout.addWidget(depth_1c_label)
        layout.addWidget(depth_3c_label)
        self.topright_groupbox.setLayout(layout)

    def init_bottom_groupbox(self):
        self.bottom_groupbox = QGroupBox('Inference')
        self.init_bottomleft_groupbox()
        self.init_bottomright_groupbox()
        layout = QHBoxLayout()
        layout.addWidget(self.bottomleft_groupbox)
        layout.addWidget(self.bottomright_groupbox)
        self.bottom_groupbox.setLayout(layout)

    def init_bottomleft_groupbox(self):
        self.bottomleft_groupbox = QGroupBox('Model Prediction')
        seg_label = QLabel(self)
        seg_label.resize(self.WIDGET_WIDTH / 2, self.WIDGET_HEIGHT)
        # read image
        qimg = self.get_qimage(IMAGE_PATH)
        pixmap = QPixmap.fromImage(qimg)
        seg_label.setPixmap(pixmap)
        layout = QHBoxLayout()
        layout.addWidget(seg_label)
        self.bottomleft_groupbox.setLayout(layout)

    def init_bottomright_groupbox(self):
        self.bottomright_groupbox = QGroupBox('Text Place')
        text_b1 = QLabel()
        text_b1.setText('Text 1')
        text_b2 = QLabel()
        text_b2.setText('Text 2')
        layout = QVBoxLayout()
        layout.addWidget(text_b1)
        layout.addWidget(text_b2)
        self.bottomright_groupbox.setLayout(layout)

    def get_qimage(self, r_path):
        img = cv2.imread(r_path)
        h, w, _ = img.shape
        bytesPerLine = 3 * w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        qimg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return qimg


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    qtmodern.styles.dark(app)
    win_modern = qtmodern.windows.ModernWindow(win)
    win_modern.show()
    sys.exit(app.exec_())