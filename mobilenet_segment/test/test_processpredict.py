"""
two bottlenecks:
1. colorEncoding
2. torch.tensor.cpu()
"""
import os
import sys
import time
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import csv
import numpy as np
import torch
from torchvision import transforms
import cv2
from img_utils import ImageLoad_cv2
from scipy.io import loadmat
from utils import colorEncode
from inference import predict, setup_model
from lib.utils import as_numpy

from profiler import profile

def preprocess():
    WIDTH = 484
    HEIGHT = 240
    ENSEMBLE_N = 3

    # GET COLOR ENCODING AND ITS INDEX MAPPING
    colors = loadmat('../data/color150.mat')['colors']
    root = '..'
    names = {}
    with open('../data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]        

    # SETUP MODEL
    cfg_path = os.path.join('..', 'config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
    #cfg_path="config/ade20k-resnet18dilated-ppm_deepsup.yaml"
    model = setup_model(cfg_path, root, gpu=0)
    model.eval()

    # GET DATA AND PROCESS IMAGE
    data = np.load(os.path.join('..', 'test_set', 'cls1_rgb.npy'))
    data = data[:, :, ::-1]
    img = ImageLoad_cv2(data, WIDTH, HEIGHT, ENSEMBLE_N, True)

    # MODEL FEED
    predictions = predict(model, img, ENSEMBLE_N, gpu = 0, is_silent = False)
    return predictions, colors, names


def process_predict_bad(scores, colors, names, is_silent):
    """
    colorEncode is used

    input:
    the predictions of model
    
    output:
    the colorize predictions
    """
    _, pred = torch.max(scores, dim=1)
    pred = as_numpy(pred.squeeze(0).cpu()) # shape of pred is (height, width)
    #The predictions for infering distance
    #seg = np.moveaxis(pred, 0, -1)
    pred = np.int32(pred)
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    if is_silent:
        return pred_color
    
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts = True)
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
    return pred_color


def process_predict_good(scores, colors, names, is_silent):
    """
    replace colorEncode by numpy way

    input:
    the predictions of model
    
    output:
    the colorize predictions
    """
    _, pred = torch.max(scores, dim=1)
    pred = as_numpy(pred.squeeze(0).cpu()) # shape of pred is (height, width)
    #The predictions for infering distance
    pred = np.int32(pred)
    pred_color = rock_the_colorencoding(pred, colors)
    if is_silent:
        return pred_color
    
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts = True)
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
    return pred_color


def rock_the_colorencoding(labelmap, colors):
    return colors[labelmap]


if __name__ == '__main__':
    # COLOR ENCODING
    predictions, colors, names = preprocess()
    print('Comparing Two Ways of Color Encoding...')
    for i in range(5):
        # bad: use colorEncode
        torch.cuda.synchronize()
        start = time.time()
        pred_color_orig = process_predict_bad(predictions, colors, names, is_silent = True)
        torch.cuda.synchronize()
        end = time.time()
        print('Original Runtime: {}s'.format(end - start))

        # good: replace by numpy lookup
        torch.cuda.synchronize()
        start = time.time()
        pred_color_gd = process_predict_good(predictions, colors, names, is_silent = True)
        torch.cuda.synchronize()
        end = time.time()
        print('Improved Runtime: {}s'.format(end - start))

    assert (pred_color_gd == pred_color_orig).all(), 'SOMETHING WRONG WITH NEW COLOR ENCODING'