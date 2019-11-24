"""
do heavy lifting job for setting up mobilenet from mobilenet_segment

PERFORMANCE COMPARISON:
1. PIL + rescaling 5: ~0.2s per interface frame
2. CV2 + adaptive rescaling: ~0.15s per interface frame

ISSUE:
1. same config, same image input, different result to Maggie's result
"""
import os
import sys
import time

import numpy as np
from scipy.io import loadmat
import csv
from torchvision import transforms
import torch
import cv2

from mobilenet_segment.inference import setup_model, predict, process_predict
from img_utils import ImageLoad_cv2
from idx_utils import create_idx_group, edit_colors_names_group
#from profiler import profile


class ModelMetaConfig:
    def __init__(self, root = os.path.join(os.getcwd(), 'mobilenet_segment') ):
        """
        ROOT is the main dir for 'data' and 'config' folder
        """
        self.RESIZE = (484, 240) # (427, 240)
        self.ENSEMBLE_N = 3
        self.setup_path(root)
        self.prepare_model()
        print('model configuration completed!')

    def setup_path(self, root):
        """
        setup index-to-color mapping (self.colors), (index+1)-to-name mapping (self.names)
        AND (original index)-to-(new index) mapping (self.idx_map)
        AND edit self.colors and self.names for grouping classes
        """
        self.ROOT = root
        self.COLOR_FILE = os.path.join(self.ROOT, 'data', 'color150.mat')
        self.COLOR2OBJ_FILE = os.path.join(self.ROOT, 'data', 'object150_info.csv')
        self.CFG_FILE = os.path.join(self.ROOT, 'config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
        # resize before input in model
        self._sanity_check()
        self.prepare_colors()
        self.prepare_names()
        # for label grouping
        self.adjust_colors_names()
        self.prepare_idx_map()

    def prepare_colors(self):
        """
        color is in BGR (NOT RBG!)
        """
        colors = loadmat(self.COLOR_FILE)['colors']
        self.colors = colors

    def prepare_names(self):
        assert 'colors' in dir(self), 'No attributes colors in ModelMetaConfig instance'
        self.names = {}
        with open(self.COLOR2OBJ_FILE) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(';')[0]
    
    def prepare_idx_map(self):
        self.idx_map = create_idx_group()
    
    def adjust_colors_names(self):
        self.colors, self.names = edit_colors_names_group(self.colors, self.names)

    def prepare_model(self):
        # set GPU mode by default
        self.model = setup_model(self.CFG_FILE, self.ROOT, gpu = 0)
        self.model.eval()

    def _sanity_check(self):
        assert os.path.isfile(self.COLOR_FILE), 'COLOR_FILE doesnt exist'
        assert os.path.isfile(self.COLOR2OBJ_FILE), 'COLOR2OBJ_FILE doesnt exist'
        assert os.path.isfile(self.CFG_FILE), 'CFG_FILE doesnt exist'
    
    def raw_predict(self, img, is_silent = True):
        width, height = self.RESIZE
        ensemble_n = self.ENSEMBLE_N
        img = ImageLoad_cv2(img, width, height, ensemble_n, is_silent = is_silent)
        #img = ImageLoad(img, width, height, ensemble_n, is_silent = is_silent)
        pred = predict(self.model, img, self.ENSEMBLE_N, is_silent = is_silent, gpu = 0)
        return pred

    def process_predict(self, pred, is_silent):
        """
        process raw model prediction into readable segmentation image
        """
        pred_color = process_predict(pred, self.colors, self.names, self.idx_map, is_silent = is_silent)
        return pred_color

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    k = 'corridor8'
    #DATA_PATH = os.path.join(os.getcwd(), 'test_cases', 'test_corridor3_rgb.jpg')
    DATA_PATH = os.path.join(os.getcwd(), 'test_cases', 'test.jpg')
    img = cv2.imread(DATA_PATH)
    #img = np.load(DATA_PATH)
    img = img[:,:,::-1]
    print('image shape = {}'.format(img.shape))
    #plt.imshow(img)
    #plt.show()
    x = ModelMetaConfig()
    for i in range(40):
        # time img process + predict
        torch.cuda.synchronize()
        start = time.time()
        pred = x.raw_predict(img, is_silent = True)
        torch.cuda.synchronize()
        end = time.time()
        print('process+predict: {}s'.format(end - start))

        torch.cuda.synchronize()
        start = time.time()
        color_pred = x.process_predict(pred, is_silent = True)
        torch.cuda.synchronize()
        end = time.time()
        print('visualize: {}s'.format(end - start))
    plt.imshow(color_pred)
    plt.show()