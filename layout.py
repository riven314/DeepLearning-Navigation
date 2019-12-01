"""
SUMMARY
create a class for layout only
layout for a new interface, it consists of 3 layers:
1. runtime for each componenets + FPS
2. scene understanding part (left: image + overlay gird, right: scene summary)
3. obstacle avoidance part (left: image, right: obj + distance)

REFERENCE:
1. create grid cell
    - https://www.delftstack.com/tutorial/pyqt5/pyqt-grid-layout/
    - https://pythonspot.com/pyqt5-grid-layout/
2. setting different text box, drop-down box: https://pythonspot.com/pyqt5-form-layout/
3. setting QGridLayout with varying height and width for each cell: https://stackoverflow.com/questions/47910192/qgridlayout-different-column-width
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
        self.title = 'Interface for Navigation'
        self.top = 100
        self.left = 100
        self.width = 1000
        self.height = 700
        self.init_window()
        
    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.init_grid_layout()
        self.show()

    def init_grid_layout(self):
        """
        set up the height and weight ratio for the three layers
        """
        grid = QGridLayout()
        # init each layers
        self.first_layer = QGroupBox('SPEED')
        self.second_layer = QGroupBox('SCENE UNDERSTANDING')
        self.third_layer = QGroupBox('OBSTACLE AVOIDANCE')
        grid.addWidget(self.first_layer, 0, 0, 1, 3)
        grid.addWidget(self.second_layer, 1, 0, 4, 6)
        grid.addWidget(self.third_layer, 5, 0, 4, 6)
        self.setLayout(grid)
        # fill up each layers
        self.fill_layers()

    def fill_layers(self):
        """
        fill up widgets for each layers
        """
        self.fill_first_layer()
        self.fill_second_layer()
        self.fill_third_layer()

    def fill_first_layer(self):
        """
        print runtime for each components + FPS (Frame Per Second)
        """
        layout = QGridLayout()
        # input time in this cell
        camera_time = QLabel('999 ms')
        model_time = QLabel('999 ms')
        fps_time = QLabel('100')
        camera_time.setFixedWidth(150)
        model_time.setFixedWidth(150)
        fps_time.setFixedWidth(150)
        # define each grid
        layout.addWidget(QLabel('Realsense Camera'), 0, 0)
        layout.addWidget(QLabel('Segmentation'), 0, 1)
        layout.addWidget(QLabel('Frame Per Second'), 0, 2)
        layout.addWidget(camera_time, 1, 0, 3, 1)
        layout.addWidget(model_time, 1, 1, 3, 1)
        layout.addWidget(fps_time,1,2, 3, 1)
        self.first_layer.setLayout(layout)
        # assign key properties
        self.camera_time = camera_time
        self.model_time = model_time
        self.fps_time = fps_time

    def fill_second_layer(self):
        """
        seg_frame: widget for printing segmentation result
        seg_summary: widget for printing objects for each grid region
        """
        # init each grid in second layer
        layout = QGridLayout()
        self.seg_widget = QGroupBox('Frame')
        self.seg_summary = QGroupBox('Summary')
        layout.addWidget(self.seg_widget, 0, 0, 1, 2)
        layout.addWidget(self.seg_summary, 0, 3, 1, 1)
        self.second_layer.setLayout(layout)
        # set up widgets for each grid
        self.fill_second_left_layer()
        self.fill_second_right_layer()

    def fill_second_left_layer(self):
        layout = QHBoxLayout()
        self.seg_frame = QLabel(self)
        layout.addWidget(self.seg_frame)
        self.seg_widget.setLayout(layout)
    
    def fill_second_right_layer(self):
        layout = QGridLayout()
        self.grid_1 = QLabel('person, floor, wall, ceiling')
        self.grid_2 = QLabel('person, floor, wall, ceiling')
        self.grid_3 = QLabel('person, floor, wall, ceiling')
        self.grid_4 = QLabel('person, floor, wall, ceiling')
        self.grid_5 = QLabel('person, floor, wall, ceiling')
        self.grid_6 = QLabel('person, floor, wall, ceiling')
        layout.addWidget(QLabel('    INDEX 1:'), 0, 0, 1, 1)
        layout.addWidget(self.grid_1, 0, 1, 1, 3)
        layout.addWidget(QLabel('    INDEX 2:'), 1, 0, 1, 1)
        layout.addWidget(self.grid_2, 1, 1, 1, 3)
        layout.addWidget(QLabel('    INDEX 3:'), 2, 0, 1, 1)
        layout.addWidget(self.grid_3, 2, 1, 1, 3)
        layout.addWidget(QLabel('    INDEX 4:'), 3, 0, 1, 1)
        layout.addWidget(self.grid_4, 3, 1, 1, 3)
        layout.addWidget(QLabel('    INDEX 5:'), 4, 0, 1, 1)
        layout.addWidget(self.grid_5, 4, 1, 1, 3)
        layout.addWidget(QLabel('    INDEX 6:'), 5, 0, 1, 1)
        layout.addWidget(self.grid_6, 5, 1, 1, 3)
        self.seg_summary.setLayout(layout)

    def fill_third_layer(self):
        """
        obj_frame: widget for printing binary mask of closest object
        obj_summary: widget for printing class name and distance of the closest object
        """
        # init each grid
        layout = QGridLayout()
        self.obj_widget = QGroupBox('Frame')
        self.obj_summary = QGroupBox('Summary')
        layout.addWidget(self.obj_widget, 0, 0, 1, 2)
        layout.addWidget(self.obj_summary, 0, 3, 1, 1)
        self.third_layer.setLayout(layout)
        # set up widget for each grid
        self.fill_third_left_layer()
        self.fill_third_right_layer()

    def fill_third_left_layer(self):
        layout = QHBoxLayout()
        self.obj_frame = QLabel(self)
        layout.addWidget(self.obj_frame)
        self.obj_widget.setLayout(layout)

    def fill_third_right_layer(self):
        layout = QGridLayout()
        self.obj_name = QLabel('other')
        self.obj_dist = QLabel('1.5 m')
        layout.addWidget(QLabel('    CLASS:'), 0, 0, 1, 1)
        layout.addWidget(self.obj_name, 0, 1, 1, 3)
        layout.addWidget(QLabel('    DISTANCE'), 1, 0, 1, 1)
        layout.addWidget(self.obj_dist, 1, 1, 1, 3)
        self.obj_summary.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Layout()
    win.show()
    sys.exit(app.exec_())