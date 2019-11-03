"""
THINGS FOR TESTING:
- mouse event get x-y coordinate

FEATURE:
1. Three plots share the output from a thread
2. Set up 2 separate loggers for different class instance

REFERENCE:
1. Is it possible to get an array from a thread in PyQt5? https://stackoverflow.com/questions/54525233/is-it-possible-to-get-an-array-from-a-thread-in-pyqt5?fbclid=IwAR2jxYj_C46ig6iQzmHLB7cZkFmVnH2GSBQW9GEK0AMxH8iqNyWldtY2Wdc
"""
import sys
import time
import logging

import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal
from polarthread import PolarThread, TrigoSticker

class Window(QDialog):
    def __init__(self):
        super().__init__()
        # setup logger
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('examples\sample.log')
        file_handler.setFormatter(formatter)
        self.logger = logging.getLogger('Window')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        # set up window
        self.title = 'PyQt5 Grid Layout'
        self.top = 100
        self.left = 100
        self.width = 100
        self.height = 600
        self.InitWindow()
        self.traces = dict()
        pg.setConfigOptions(antialias = True)
        


    def InitWindow(self):
        #self.setWindowIcon(QtGui.QIcon('icon.png))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.gridLayoutCreation()

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.groupBox) # what it is doing?
        self.setLayout(vboxLayout)
        self.logger.info('window setup! ready to rock!')
        self.show()

    def gridLayoutCreation(self): 
        """
        Create the widget corresponding to the plot
        """
        self.groupBox = QGroupBox("Grid Layout Example")
        gridLayout = QGridLayout()
        self.guiplot_1 = pg.PlotWidget()
        self.guiplot_2 = pg.PlotWidget()
        self.guiplot_3 = pg.PlotWidget()
        gridLayout.addWidget(self.guiplot_1, 0, 8, 8, 12)
        gridLayout.addWidget(self.guiplot_2, 8, 8, 8, 12)
        gridLayout.addWidget(self.guiplot_3, 16, 8, 8, 12)
        self.groupBox.setLayout(gridLayout)

        gridLayout.addWidget(QLabel('Tempo'), 0,0)
        self.timeEdit = QLineEdit('')                       # time <-> timeEdit
        gridLayout.addWidget(self.timeEdit, 1,0)            # time <-> timeEdit

    def plotar(self, s, f_type):
        if f_type == 's':                                    
            self.guiplot_1.clear()
            self.guiplot_1.plot(s)
            self.logger.info('sine f is plotted!')
        elif f_type == 'c':
            self.guiplot_2.clear()
            self.guiplot_2.plot(s)
            self.logger.info('cosine f is plotted!')
        else:
            self.guiplot_3.clear()
            self.guiplot_3.plot(s)
            self.logger.info('tan f is plotted!')
        
    def teste(self):
        self.get_thread = PolarThread()
        # below only run once
        self.get_thread.signal.connect(lambda z: self.display_graph(z, 's'))
        self.get_thread.signal.connect(lambda z: self.display_graph(z, 'c'))
        self.get_thread.signal.connect(lambda z: self.display_graph(z, 't'))
        self.logger.info('thread set up complete! ready to rock!')
        self.get_thread.start()

    def display_graph(self, sticker, f_type):                              # <--- +++
        """ Here is your `self.s` array. 
            Draw a graph in real time without blocking the graphical interface.
        """
        # print("\n Here is your `self.s` array. \n", self_s)
        if f_type == 's':
            self.plotar(sticker.s, 's')
        elif f_type == 'c':
            self.plotar(sticker.c, 'c')
        elif f_type == 't':
            self.plotar(sticker.t, 't')
        else:
            print('SOMETHING WRONG!')
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Window()
    form.show()
    form.teste()
    sys.exit(app.exec_())