"""
implement an obstacle avoidance algorithm

input:
1. prediced labels: pred  (720X1280)
2. 1d depth image related to the rgb image (720X1280)

output:
1. 'ALARM' or not
2. how far is the nearest object (if there is 'alarm') 
3. the label of nearest object
4. vasulize the object

hyperparameters:
1. central part: [600:680], i have tested many ranges, this one performs best; it should be changed if you change the size of images
2. depth threshold: i set 2000
3. threshold of the number of pixels: i set 500
4: object_classes = [2,3,4,5,6]
5. label_dict = { 1:'floor',2: 'furniture', 3:'objects',  4: 'person', 5: 'wall', 6:'door', 7:'ceiling'}

note:
we should make sure the size of pred and d1 are the same
"""
import os
import sys
import time
import copy
import re

import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import skimage.io as io
from skimage import measure, color

label_dict = { 1: 'floor', 2: 'furniture', 3:'objects',  4: 'person', 5: 'wall', 6:'door', 7:'ceiling'}
#names = {1: 'wall', 2: 'floor', 3: 'plant', 4: 'ceiling', 5: 'furniture', 6: 'person', 7: 'door', 8: 'objects'}

# generate a test case with depth + segmentation shape is the same

x = np.random.randint(0, 255, size = (720, 1280, 3), dtype = np.uint8)
for i in range(5):
    start = time.time()
    cv2.resize(x, (427, 240), interpolation = cv2.INTER_LINEAR)
    end = time.time()
    print('run time: {} s'.format(end - start))
