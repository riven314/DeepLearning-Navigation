"""
libjpeg-turbo is installed in "C:/libjpeg-turbo-gcc"

REFERENCE:
1. PyTurboJPEG installation https://pypi.org/project/PyTurboJPEG/?fbclid=IwAR2tbXI5kvr7juvm63JazeHZpSXkn5gaRXFzSz5Z_gfqD20GS-fn9F9KSIk

LOG
[20/10/2019]
still can't use PyTurboJPEG due to library path can't find required file
"""
import timeit

import numpy as np
import cv2
from turbojpeg import TurboJPEG

# installed in /mnt/c/libjpeg-turbo-gcc/lib/ 
jpeg = TurboJPEG(r'C:/libjpeg-turbo-gcc/lib/libturbojpeg.dll.a')

def imencode(img, is_cv2 = True):
    if is_cv2:
        _, jpeg = cv2.imencode('.bmp', img)
    else:
        jpeg = jpeg.encode(img)
    return jpeg

def time_imencode(n = 100, is_cv2 = True):
    img = np.random.randn(720, 1280, 3)
    t = timeit.timeit(imencode(img, is_cv2), number = n)
    print('run time: {} s'.format(t))

if __name__ == '__main__':
    print('RUN CV2 ENCODING...')
    time_imencode(n = 100, is_cv2 = True)
    print('RUN TURBO ENCODING...')
    time_imencode(n = 100., is_cv2 = False)

