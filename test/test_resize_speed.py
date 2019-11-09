"""
CANDIDATES:
1. cv2 (realsense env)
2. PIL (realsense env)
3. pillow-simd (tmp in labelme env)


SPEED:
1. cv2 ~ 0.006 s
2. PIL ~ 0.0254 s

"""
import time
import numpy as np

import cv2
from PIL import Image
import torch

from profiler import profile

def pil_resize():
    data = np.random.randint(0, 255, size = (1280, 720, 3), dtype = np.uint8)
    img = Image.fromarray(data)
    resized = img.resize((484, 240), resample = Image.BILINEAR)
    #print('final img shape = {}'.format(np.array(resized).shape))

def cv2_resize():
    data = np.random.randint(0, 255, size = (1280, 720, 3), dtype = np.uint8)
    resized = cv2.resize(data, (484, 240), interpolation = cv2.INTER_LINEAR)
    #print('final img shape = {}'.format(resized.shape))

def time_func(f, n = 50):
    t_ls = []
    for i in range(n):
        start = time.time()
        f()
        end = time.time()
        t_ls.append(end - start)
    t_ls = np.array(t_ls)
    ans = t_ls.mean()
    print('avg run time = {} s'.format(ans))

def torch_resize():
    torch.nn.Upsample(size = (484, 240), mode = 'bilinear')(torch.rand(1, 3, 1280, 720))

def time_torch(n = 50):
    t_ls = []
    for i in range(n):
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()
        torch_resize()
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        t_ls.append(t)
    t_ls = np.array(t_ls)
    ans = t_ls.mean() / 1000
    print('avg run time = {} s'.format(ans))

if __name__ == '__main__':
    #time_torch(n = 1000)
    print('cv2 resize...')
    time_func(cv2_resize, n = 100)
    print('pil resize...')
    time_func(pil_resize, n = 100)
    #pil_resize()
    #cv2_resize()
    #pil_resize()
