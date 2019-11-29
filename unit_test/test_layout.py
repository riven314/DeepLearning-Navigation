"""
testing layout.py
"""
import os
import sys
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import cv2
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtWidgets import QApplication
import qtmodern.styles
import qtmodern.windows

from layout import Layout
from pyqt_utils import convert_qimg

IMG_1_PATH = os.path.join('..', 'test_cases', 'test3_pred_rgb.jpg')
IMG_2_PATH = os.path.join('..', 'test_cases', 'binary_mask.png')

class TestLayout(Layout):
    def __init__(self):
        super().__init__()
        self.load_img()
        self.set_pixmap()
    
    def load_img(self):
        data1 = cv2.imread(IMG_1_PATH)
        data2 = cv2.imread(IMG_2_PATH)
        self.data1 = data1.copy()
        self.data2 = data2.copy()
        
    def set_pixmap(self):
        qimg1 = convert_qimg(self.data1, win_width = 620, win_height = 360)        
        qimg2 = convert_qimg(self.data2, win_width = 620, win_height = 360)
        self.seg_frame.setPixmap(QPixmap.fromImage(qimg1))
        self.obj_frame.setPixmap(QPixmap.fromImage(qimg2))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TestLayout()
    qtmodern.styles.dark(app)
    win_modern = qtmodern.windows.ModernWindow(win)
    win_modern.show()
    sys.exit(app.exec_())
    
