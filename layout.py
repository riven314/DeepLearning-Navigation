"""
SUMMARY
create a class for layout only
layout for a new interface, it consists of 3 layers:
1. runtime for each componenets + FPS
2. scene understanding part (left: image + overlay gird, right: scene summary)
3. obstacle avoidance part (left: image, right: obj + distance)

TO BE SETTLE:
1. adjust the size for each layer and its components

REFERENCE:
1. create grid cell
    - https://www.delftstack.com/tutorial/pyqt5/pyqt-grid-layout/
    - https://pythonspot.com/pyqt5-grid-layout/
2. setting different text box, drop-down box: https://pythonspot.com/pyqt5-form-layout/
"""
import os
import sys

from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QLabel,  QLineEdit
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QWidget


class Layout(QWidget):
    def __init__(self):
        # set up window
        super().__init__()
        self.title = 'Navigation Tool'
        self.top = 100
        self.left = 100
        self.width = 1280
        self.height = 1280
        self.init_window()
        
    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.wrap_three_layers()
        self.show()

    def wrap_three_layers(self):
        self.init_first_layer()
        self.init_second_layer()
        self.init_third_layer()
        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.first_layer)
        vbox_layout.addWidget(self.second_layer)
        vbox_layout.addWidget(self.third_layer)
        self.setLayout(vbox_layout)

    def init_first_layer(self):
        """
        print runtime for each components + FPS (Frame Per Second)
        """
        self.first_layer = QGroupBox("SPEED")
        layout = QGridLayout()
        # input time in this cell
        camera_time = QLineEdit('--NA--')
        model_time = QLineEdit('--NA--')
        fps_time = QLineEdit('--NA--')
        # define each grid
        layout.addWidget(QLabel('Camera'), 0, 0)
        layout.addWidget(QLabel('Model'), 0, 1)
        layout.addWidget(QLabel('FPS'), 0, 2)
        layout.addWidget(camera_time, 1, 0)
        layout.addWidget(model_time, 1, 1)
        layout.addWidget(fps_time,1,2)
        self.first_layer.setLayout(layout)
        
        # assign key properties
        self.camera_time = camera_time
        self.model_time = model_time
        self.fps_time = fps_time

    def init_second_layer(self):
        """
        layer for scene understanding
        left: segmentation result overlayed with a grid
        right: summary text for each grid
        """
        self.second_layer = QGroupBox('SCENE UNDERSTANDING')
        self.init_second_left_layer()
        self.init_second_right_layer()
        layout = QGridLayout()
        layout.addWidget(self.second_left_layer, 0, 0)
        layout.addWidget(self.second_right_layer, 0, 1)
        self.second_layer.setLayout(layout)

    def init_second_left_layer(self):
        """
        [scene understanding] -- set up segmentation output
        """
        self.second_left_layer = QGroupBox('Segmentation')
        layout = QGridLayout()
        seg_out = QLabel(self)
        layout.addWidget(seg_out, 0, 0)
        self.second_left_layer.setLayout(layout)

        # assign key properties
        self.seg_out = seg_out
    
    def init_second_right_layer(self):
        """
        [scene understanding] -- set up summary for each grid
        """
        self.second_right_layer = QGroupBox('Grid Summary')
        layout = QGridLayout()
        grid_1 = QLineEdit('INDEX 1')
        grid_2 = QLineEdit('INDEX 2')
        grid_3 = QLineEdit('INDEX 3')
        grid_4 = QLineEdit('INDEX 4')
        grid_5 = QLineEdit('INDEX 5')
        grid_6 = QLineEdit('INDEX 6')
        layout.addWidget(QLabel('INDEX 1:'), 0, 0)
        layout.addWidget(grid_1, 0, 1)
        layout.addWidget(QLabel('INDEX 2:'), 1, 0)
        layout.addWidget(grid_2, 1, 1)
        layout.addWidget(QLabel('INDEX 3:'), 2, 0)
        layout.addWidget(grid_3, 2, 1)
        layout.addWidget(QLabel('INDEX 4:'), 3, 0)
        layout.addWidget(grid_4, 3, 1)
        layout.addWidget(QLabel('INDEX 5:'), 4, 0)
        layout.addWidget(grid_5, 4, 1)
        layout.addWidget(QLabel('INDEX 6:'), 5, 0)
        layout.addWidget(grid_6, 5, 1)
        self.second_right_layer.setLayout(layout)

        # assign key properties
        self.grid_1 = grid_1
        self.grid_2 = grid_2
        self.grid_3 = grid_3
        self.grid_4 = grid_4
        self.grid_5 = grid_5
        self.grid_6 = grid_6

    def init_third_layer(self):
        """ 
        layer for obstacle avoidance
        left: binary image showing close object
        right: (object, distance)
        """
        self.third_layer = QGroupBox('OBSTACLE AVOIDANCE')
        self.init_third_left_layer()
        self.init_third_right_layer()
        layout = QGridLayout()
        layout.addWidget(self.third_left_layer, 0, 0)
        layout.addWidget(self.third_right_layer, 0, 1)
        self.third_layer.setLayout(layout)

    def init_third_left_layer(self):
        """
        [scene understanding] -- set up segmentation output
        """
        self.third_left_layer = QGroupBox('Closest Object')
        layout = QGridLayout()
        obj_mask = QLabel(self)
        layout.addWidget(obj_mask, 0, 0)
        self.third_left_layer.setLayout(layout)

        # assign key properties
        self.obj_mask = obj_mask
    
    def init_third_right_layer(self):
        """
        [scene understanding] -- set up summary for each grid
        """
        self.third_right_layer = QGroupBox('Its Class and Distance')
        layout = QGridLayout()
        obj_name = QLineEdit('OBJECT CLASS')
        obj_dist = QLineEdit('OBJECT DISTANCE')
        layout.addWidget(QLabel('CLASS:'), 0, 0)
        layout.addWidget(obj_name, 0, 1)
        layout.addWidget(QLabel('DISTANCE'), 1, 0)
        layout.addWidget(obj_dist, 1, 1)
        self.third_right_layer.setLayout(layout)

        # assign key properties
        self.obj_name = obj_name
        self.obj_dist = obj_dist


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Layout()
    win.show()
    sys.exit(app.exec_())