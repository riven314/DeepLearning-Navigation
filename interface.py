import os
import sys

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtWidgets import QApplication
import qtmodern.styles
import qtmodern.windows

from layout import Layout
from pyqt_utils import convert_qimg
from scene_summary import get_names, create_grid, scene_summarize
from thread_utils import FrameStore, FrameThread
from obj_avoidance import run_avoidance

class SimplifyInteface(Layout):
    def __init__(self):
        super().__init__()
        # setup for scene understanding
        self.load_lightbulb()
        self.mat = create_grid(h = 240, w = 427)
        self.names = get_names()
        self.init_thread()
    
    def load_lightbulb(self):
        RED_PATH = os.path.join('images', 'red.jpg')
        GREEN_PATH = os.path.join('images', 'green.jpg')
        assert os.path.isfile(RED_PATH), '[ERROR] Path not exist: {}'.format(RED_PATH)
        assert os.path.isfile(GREEN_PATH), '[ERROR] Path not exist: {}'.format(GREEN_PATH)
        red = cv2.imread(RED_PATH)
        green = cv2.imread(GREEN_PATH)
        self.red_qimg = convert_qimg(red, win_width = 50, win_height = 50)
        self.green_qimg = convert_qimg(green, win_width = 50, win_height = 50)

    def init_thread(self):
        self.f_thread = FrameThread()
        # retrieve class mapping, color mapping and camera scale
        self.seg_names = self.f_thread.model_config.names
        self.seg_colors = self.f_thread.model_config.colors
        self.depth_scale = self.f_thread.rs_camera.depth_scale
        # connect thread to different slots
        self.f_thread.frame_signal.connect(lambda frame_store: self.update_first_layer(frame_store))
        self.f_thread.frame_signal.connect(lambda frame_store: self.update_second_layer(frame_store))
        self.f_thread.frame_signal.connect(lambda frame_store: self.update_third_layer(frame_store))
        self.f_thread.start()
    
    def update_first_layer(self, frame_store):
        """
        update different runtime and FPS
        """
        self.camera_time.setText('{0:.1f} ms'.format(frame_store.camera_time * 1000))
        self.model_time.setText('{0:.1f} ms'.format(frame_store.model_time * 1000))
        self.fps_time.setText('{0:.1f}'.format(frame_store.fps_time))

    def update_second_layer(self, frame_store):
        """
        update segmentation result and scene summary
        """
        # update segmentation result
        qimg = convert_qimg(frame_store.pred_rgb, win_width = 620, win_height = 360)
        self.seg_frame.setPixmap(QPixmap.fromImage(qimg))
        # update scene summary
        grid_dict = scene_summarize(frame_store.pred_idx, 
                                    self.mat, self.names,
                                    threshold = 900)
        self.update_scene_summary(grid_dict)
    
    def update_scene_summary(self, grid_dict):
        for i, obj_ls in grid_dict.items():
            txt = ', '.join(obj_ls)
            q_label = getattr(self, 'grid_{}'.format(i + 1))
            q_label.setText(txt)

    def update_third_layer(self, frame_store):
        """
        update obstacle avoidance (frame + summary)
        """
        obj_tup, obj_img = run_avoidance(frame_store.d1_img, frame_store.pred_idx, depth_threshold = 6)
        # convert to uint8 is important!
        obj_img = np.uint8(obj_img)
        qimg = convert_qimg(obj_img, win_width = 620, win_height = 360, is_gray = True)
        # update frame on left
        self.obj_frame.setPixmap(QPixmap.fromImage(qimg))
        # update summary on right
        if obj_tup[1] is None:
            self.obj_name.setText('NA')
            self.obj_dist.setText('NA')
            # raise green alarm
            self.lightbulb.setPixmap(QPixmap.fromImage(self.green_qimg))
        else:
            obj_name = self.names[obj_tup[1] + 1]
            # translate pixel value to meter
            meter = obj_tup[2] * self.depth_scale
            self.obj_name.setText(obj_name)
            self.obj_dist.setText('{0:.1f} cm'.format(meter * 100))
            # raise red alarm
            self.lightbulb.setPixmap(QPixmap.fromImage(self.red_qimg))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SimplifyInteface()
    qtmodern.styles.dark(app)
    win_modern = qtmodern.windows.ModernWindow(win)
    win_modern.show()
    sys.exit(app.exec_())
