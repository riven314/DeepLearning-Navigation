"""
testing layout.py

verify that:
1. frame can be updated
2. text can be updated
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
        self.update_pixmap()
        self.update_text()
    
    def load_img(self):
        data1 = cv2.imread(IMG_1_PATH)
        data2 = cv2.imread(IMG_2_PATH)
        self.data1 = data1.copy()
        self.data2 = data2.copy()
        
    def update_pixmap(self):
        qimg1 = convert_qimg(self.data1, win_width = 620, win_height = 360)        
        qimg2 = convert_qimg(self.data2, win_width = 620, win_height = 360)
        self.seg_frame.setPixmap(QPixmap.fromImage(qimg1))
        self.obj_frame.setPixmap(QPixmap.fromImage(qimg2))
    
    def update_text(self):
        self.camera_time.setText('144 ms')
        self.grid_1.setText('person, person, person')
        self.obj_dist.setText('1.44 m')

    def update_lightbulb(self):
        from pyqt_utils import convert_qimg
        import cv2
        from PyQt5.QtGui import QPixmap

        img = cv2.imread(os.path.join('..', 'images', 'green.jpg'))
        qimg = convert_qimg(img, win_width = 50, win_height = 50)
        self.lightbulb.setPixmap(QPixmap.fromImage(qimg))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TestLayout()
    win.update_lightbulb()
    qtmodern.styles.dark(app)
    win_modern = qtmodern.windows.ModernWindow(win)
    win_modern.show()
    sys.exit(app.exec_())
    
