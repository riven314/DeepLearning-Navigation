"""
Not used, only for backup
"""
import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, Qt, pyqtSignal

class RealsenseThread(QThread):
    # set up signal 
    changePixmap = pyqtSignal(QImage)
    # set up camera
    resolution = (1280, 720)
    rs_camera = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    def run(self):
        #ptvsd.debug_this_thread()
        while True:
            frames = self.rs_camera.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            rgb_frame = frames.get_color_frame() # uint 8
            depth_frame = frames.get_depth_frame() # unit 8
            color_image = np.asanyarray(rgb_frame.get_data())
            #depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            display_image = np.concatenate((color_image, depth_colormap), axis=1)
            qimg = self.convert_qimg(color_image)
            self.changePixmap.emit(qimg)

    def convert_qimg(self, frame, win_width = 640, win_height = 480):
        """
        convert from cv2 frame to QImage frame
        """
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        byte_per_line = c * w
        img = QtGui.QImage(rgb_img.data, w, h, byte_per_line, QtGui.QImage.Format_RGB888)\
                    .scaled(win_width, win_height, Qt.KeepAspectRatio)
        return img


class WebcamThread(QThread):
    # a signal called "changePixmap" that take an argument QImage
    # QImage is an image representation that allows direct access to pixel data
    changePixmap = pyqtSignal(QImage)
    cap = cv2.VideoCapture(0) # uint 8

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                img = self.convert_qimg(frame)
                self.changePixmap.emit(img)

    def convert_qimg(self, frame, win_width = 640, win_height = 480):
        """
        convert from cv2 frame to QImage frame
        """
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        byte_per_line = c * w
        img = QtGui.QImage(rgb_img.data, w, h, byte_per_line, QtGui.QImage.Format_RGB888)\
                    .scaled(win_width, win_height, Qt.KeepAspectRatio)
        return img