"""
REFERENCE:
1. pyqt5 doc on customising signals: https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html
2. print message on console when using pyqt5: https://forums.autodesk.com/t5/motionbuilder-forum/pyqt-pyside-event-handlers-don-t-print-to-console/td-p/7058029
3. advanced logging youtube: https://www.youtube.com/watch?v=jxmzY9soFXg&t=1042s
4. how to setup logging in pyqt5: https://stackoverflow.com/questions/28655198/best-way-to-display-logs-in-pyqt
5. python operator overloading: https://www.youtube.com/watch?v=9wd50TKv_OQ
6. python method overloading and method overriding: https://www.youtube.com/watch?v=CcTzTuIsoFk
"""
import os
import logging
from PyQt5.QtCore import QObject, pyqtSignal

logging.basicConfig(filename = os.path.join('test', 'sample.log'), 
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(name)s:%(message)s')

class Foo(QObject): # new signals must inherent from QObject
    trigger = pyqtSignal()

    def connect_and_emit_trigger(self):
        self.trigger.connect(self.handle_trigger)
        logging.debug('hello world. It is connected')
        self.trigger.emit()

    def handle_trigger(self):
        logging.debug('hello world. It is triggered')
        logging.debug('hello world. It is triggered')


if __name__ == '__main___':
    x = Foo()
    x.handle_trigger()
    x.connect_and_emit_trigger()