import os
import sys
import time
root_path = os.path.join(os.getcwd(), '..')
seg_module_path = os.path.join(os.getcwd(), '..', 'mobilenet_segment')
sys.path.append(root_path)
sys.path.append(seg_module_path)

import numpy as np
from scipy.io import loadmat
import csv
from torchvision import transforms

from webcam_test import ImageLoad, setup_model, predict, process_predict
from config.defaults import _C as cfg

#Define the color dict
COLOR_PLATE_PATH = os.path.join('..', 'mobilenet_segment', 'data', 'color150.mat')
PRINT_PATH = os.path.join('..', 'mobilenet_segment', 'data', 'object150_info.csv')
DATA_PATH = os.path.join('..', 'mobilenet_segment', 'test_set', 'cls1_rgb.npy')
ROOT = os.path.join(os.getcwd(), '..', 'mobilenet_segment')
WIDTH = 640
HEIGHT = 360
RESIZE_NUM = 3

colors = loadmat(COLOR_PLATE_PATH)['colors']
names = {}
with open(PRINT_PATH) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0] 


img = np.load(DATA_PATH)
img = img[:,:,::-1]
cfg_path = os.path.join('..', 'mobilenet_segment', 'config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
#cfg_path="config/ade20k-resnet18dilated-ppm_deepsup.yaml"

import matplotlib.pyplot as plt

model = setup_model(cfg_path, ROOT, gpu=0)
for i in range(10):
    Iamge = ImageLoad(img, WIDTH, HEIGHT)
    start = time.time()
    predictions = predict(model, Iamge, RESIZE_NUM ,gpu=0)
    end = time.time()
    print('runtime = {} s'.format(end - start))
    seg,pred_color = process_predict(predictions, colors, names)
plt.imshow(pred_color)
plt.show()
