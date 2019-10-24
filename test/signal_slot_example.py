"""
REFERENCE:
- PyQt5 Lesson 5 Signals and Slots https://www.youtube.com/watch?v=h09_iyXR81Y&list=PLZocUikpczs-Yud2lyFpSNQOvxuPUVBDp&index=5
"""
import sys
from PyQt5 import QtWidgets

class Window(QtWidgets.QWidget):
    CLICKED = 'I have been clicked'
    NOT_CLICKED = 'I have not been clicked yet'    

    def __init__(self):
        # construct our parent class
        super().__init__()
        self.init_ui()
        

    def init_ui(self):
        self.b = QtWidgets.QPushButton('Push Me')
        self.l = QtWidgets.QLabel(self.NOT_CLICKED)

        # LAYOUT DESIGN        
        h_box = QtWidgets.QHBoxLayout()
        # place label on a box
        h_box.addStretch()
        h_box.addWidget(self.l)
        h_box.addStretch() # add stretch twice so as to place label at center when window expanded
        # place a button on a box
        v_box = QtWidgets.QVBoxLayout()
        v_box.addWidget(self.b)
        # add h_box at bottom
        v_box.addLayout(h_box)
        self.setLayout(v_box)
        
        # DEFINE SIGNAL
        self.b.clicked.connect(self.btn_clicked) # do btn_clicked when signal "b.clicked" is activated

        self.setWindowTitle('PyQt5 Lesson 5')
        self.show()

    def btn_clicked(self):
        if self.l.text() == self.CLICKED:
            self.l.setText(self.NOT_CLICKED)
        else:
            self.l.setText(self.CLICKED)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    a_window = Window()
    sys.exit(app.exec_())

