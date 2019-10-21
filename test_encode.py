"""
libjpeg-turbo is installed in "C:/libjpeg-turbo"

REFERENCE:
1. PyTurboJPEG installation https://pypi.org/project/PyTurboJPEG/?fbclid=IwAR2tbXI5kvr7juvm63JazeHZpSXkn5gaRXFzSz5Z_gfqD20GS-fn9F9KSIk

LOG
[20/10/2019]
still can't use PyTurboJPEG due to library path can't find required file
"""
import timeit
import time

import numpy as np
import cv2
from turbojpeg import TurboJPEG

# installed in /mnt/c/libjpeg-turbo-gcc/lib/ 
jpeg = TurboJPEG()

def imencode(img, is_cv2 = True):
    if is_cv2:
        _, jpg = cv2.imencode('.jpg', img)
    else:
        jpg = jpeg.encode(img)
    return jpg

def time_imencode(n = 100, is_cv2 = True):
    # imitate the concatenation
    img = np.random.randn(720 , 1280 * 2, 3)
    t_ls = []
    for i in range(n):
        start = time.time()
        imencode(img, is_cv2)
        end = time.time()
        t_ls.append(end - start)
    t_ls = np.array(t_ls)
    print('run time: {} s'.format(t_ls.mean()))

if __name__ == '__main__':
    print('RUN CV2 ENCODING...')
    time_imencode(n = 100, is_cv2 = True)
    print('RUN TURBO ENCODING...')
    time_imencode(n = 100, is_cv2 = False)

