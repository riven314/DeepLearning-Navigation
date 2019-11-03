import logging
import time

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal


class TrigoSticker():
    def __init__(self):
        """
        storing sine(s), cosine(c) and tan(t) function points
        """
        self.s = None
        self.c = None
        self.t = None

class PolarThread(QThread):
    # a signal that emit TrigoSticker instance
    signal = pyqtSignal(TrigoSticker)

    def __init__(self, parent = None):
        super().__init__()
        # setup logger
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('examples\sample.log')
        file_handler.setFormatter(formatter)
        self.logger = logging.getLogger('QThread')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.s = None
        self.phase = 0
        self.sticker = TrigoSticker()

    def update(self):
        self.t = np.arange(0, 3., 0.01)
        # 3 different plots
        self.sticker.s = np.sin(2 * np.pi * self.t + self.phase)
        self.sticker.c = np.cos(2 * np.pi * self.t + self.phase)
        self.sticker.t = np.tan(2 * np.pi * self.t + self.phase)
        self.logger.info(time.time())
        self.phase += 0.1
        QThread.msleep(200) # 2000 ms (2s)
    
    def run(self):
        for i in range(100):
            self.logger.info('running qthread!')
            self.update()
            self.signal.emit(self.sticker)