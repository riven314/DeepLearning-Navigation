"""
test using logger in pyqt5 modules

QUESTION:
1. where does signal.emit() activated?
2. any way to dump logger message into a log file?
"""
import sys
from PyQt5 import QtWidgets
import logging

class QTextEditLogger(logging.Handler):
    # what is parent??
    def __init__(self, parent): 
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
    
    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class MyDialog(QtWidgets.QDialog, QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        logTextBox = QTextEditLogger(self)
        # You can format what is printed to text box
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logTextBox.setFormatter(formatter)
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logTextBox)
        file_handler = logging.FileHandler('test\sample.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._button = QtWidgets.QPushButton(self)
        self._button.setText('Test Me')

        layout = QtWidgets.QVBoxLayout()
        # Add the new logging box widget to the layout
        layout.addWidget(logTextBox.widget)
        layout.addWidget(self._button)
        self.setLayout(layout)

        # Connect signal to slot
        self._button.clicked.connect(self.test)

    def test(self):
        self.logger.debug('damn, a bug')
        self.logger.info('something to remember')
        self.logger.warning('that\'s not right')
        self.logger.error('foobar')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dlg = MyDialog()
    dlg.show()
    dlg.raise_()
    sys.exit(app.exec_())