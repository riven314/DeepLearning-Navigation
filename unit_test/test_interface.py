"""
test the following:
model + scene understanding + interface
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
from scene_summary import get_names, create_grid, scene_summarize
from simplify_thread_utils import FrameStore, FrameThread

class SimplifyInteface(Layout):
    def __init__(self):
        super().__init__()
        # setup for scene understanding
        self.mat = create_grid(h = 240, w = 427)
        self.names = get_names()
        self.init_thread()
    
    def init_thread(self):
        self.f_thread = FrameThread()
        self.seg_names = self.f_thread.model_config.names
        self.seg_colors = self.f_thread.model_config.colors
        self.f_thread.frame_signal.connect(lambda frame_store: self.update_first_layer(frame_store))
        self.f_thread.frame_signal.connect(lambda frame_store: self.update_second_layer(frame_store))
        self.f_thread.start()
    
    def update_first_layer(self, frame_store):
        """
        update different runtime and FPS
        """
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SimplifyInteface()
    qtmodern.styles.dark(app)
    win_modern = qtmodern.windows.ModernWindow(win)
    win_modern.show()
    sys.exit(app.exec_())




