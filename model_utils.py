"""
do heavy lifting job for setting up mobilenet from mobilenet_segment

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

from mobilenet_segment.webcam_test import ImageLoad, setup_model, predict, process_predict
#from config.defaults import _C as cfg

class ModelMetaConfig:
    def __init__(self, root = os.path.join(os.getcwd(), 'mobilenet_segment') ):
        """
        ROOT is the main dir for 'data' and 'config' folder
        """
        self.ROOT = root
        self.COLOR_FILE = os.path.join(self.ROOT, 'data', 'color150.mat')
        self.COLOR2OBJ_FILE = os.path.join(self.ROOT, 'data', 'object150_info.csv')
        self.CFG_FILE = os.path.join(self.ROOT, 'config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
        # resize before input in model
        self.RESIZE = (484, 240) # (width, height)  
        self.ENSEMBLE_N = 3
        # configure names, colors and model
        self._sanity_check()
        self.prepare_colors()
        self.prepare_names()
        self.prepare_model()
        print('model configuration completed!')

    def prepare_colors(self):
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

    def prepare_model(self):
        # set GPU mode by default
        self.model = setup_model(self.CFG_FILE, self.ROOT, gpu = 0)

    def _sanity_check(self):
        assert os.path.isfile(self.COLOR_FILE), 'COLOR_FILE doesnt exist'
        assert os.path.isfile(self.COLOR2OBJ_FILE), 'COLOR2OBJ_FILE doesnt exist'
        assert os.path.isfile(self.CFG_FILE), 'CFG_FILE doesnt exist'
        
    def raw_predict(self, img):
        """
        do model prediction, output raw model prediction

        input:
            img -- np array
        output:
            pred -- np array, raw model prediction (with proability and class index)
        """
        width, height = self.RESIZE
        img = ImageLoad(img, width, height)
        pred = predict(self.model, img, self.ENSEMBLE_N, gpu = 0)
        return pred

    def process_predict(self, pred):
        """
        process raw model prediction into readable segmentation image
        """
        _, pred_color = process_predict(pred, self.colors, self.names)
        return pred_color

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    DATA_PATH = os.path.join(os.getcwd(), 'mobilenet_segment', 'test_set', 'cls1_rgb.npy')
    img = np.load(DATA_PATH)
    img = img[:,:,::-1]
    plt.imshow(img)
    plt.show()
    x = ModelMetaConfig()
    pred = x.raw_predict(img)
    color_pred = x.process_predict(pred)
    plt.imshow(color_pred)
    plt.show()
    
