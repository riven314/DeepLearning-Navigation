"""
implement an obstacle avoidance algorithm

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
from skimage import measure

# by maggir
#label_dict = { 1: 'floor', 2: 'furniture', 3:'objects',  4: 'person', 5: 'wall', 6:'door', 7:'ceiling'}  
# note that colors[key] --> names[key+1] (by me)
#label_dict = {1: 'wall', 2: 'floor', 3: 'plant', 4: 'ceiling', 5: 'furniture', 6: 'person', 7: 'door', 8: 'objects'}

def run_avoidance(d1_img, seg_idx, depth_threshold = 8, visible_width = 90):
    """
    input:
        d1_img -- np array, 1 channel depth
        seg_idx -- np array, segmentation output with index only
        depth_threshold -- int, for apply masking by depth
        visible_width -- 90, number of pixel up front you wanna be visible
        ** make sure d1_img, seg_idx have same size
    output:
        obj_tup -- tuple, (class idx, distance, (x, y))
        obj_img -- np array -- image showing single object only
    """
    # omit floor (idx = 1, name = 2) and ceiling (idx = 3, name = 4)
    seg_idx[seg_idx == 1] = 0
    seg_idx[seg_idx == 3] = 0
    # find connected components (index per instance)
    inst_idx = measure.label(seg_idx, connectivity = 2)
    # get closest object info
    obj_tup  = get_obj_info(d1_img, seg_idx, inst_idx, depth_threshold = depth_threshold, visible_width = visible_width)
    # get the closest object image
    obj_img = get_obj_img(d1_img, inst_idx, obj_tup[0])
    return obj_tup, obj_img


def get_obj_info(d1_img, seg_idx, inst_idx, depth_threshold = 8, visible_width = 90):
    """
    input:
        d1_img -- np array, 1 channel depth
        seg_idx -- np array, segmentation index map
        inst_idx -- np array, output from connected components
        depth_threshold -- int, for apply masking by depth
        visible_width -- 90, number of pixel up front you wanna be visible
        ** make sure d1_img, seg_idx have same size
    output:
        (min_inst_idx, min_cls_idx, min_dist) -- (instance index, class index, its distance)
    """
    # apply filter on distance and angle
    _, w = seg_idx.shape
    lower_limit = int(w/2 - visible_width/2)
    upper_limit = int(w/2 + visible_width/2)
    filter_inst_idx = (d1_img < depth_threshold) * inst_idx
    filter_inst_idx[:, :lower_limit] = 0
    filter_inst_idx[:, upper_limit:] = 0
    # find closeset distance
    min_inst_idx, min_cls_idx, min_dist = None, None, float('inf')
    for idx in np.unique(filter_inst_idx):
        idx_locs = np.where(inst_idx == idx)
        loc = (idx_locs[0][0], idx_locs[1][0])
        cls_idx = seg_idx[loc[0], loc[1]]
        # skip 0 index (background)
        if cls_idx == 0:
            continue
        depth = d1_img[idx_locs]
        # remove noise in depth map
        depth[depth == 0] = depth.max()
        dist = np.sort(depth, axis = None)[:10].mean()
        if dist < min_dist:
            min_inst_idx = idx
            min_cls_idx = cls_idx
            min_dist = dist
    return min_inst_idx, min_cls_idx, min_dist


def get_obj_img(d1_img, inst_idx, target_idx):
    """
    visualise the whole closest object 

    input:
        d1_img -- np array, 1 channel depth
        seg_idx -- np array, segmentation output with index only
        target_idx -- int, target index of the instance
    output:
        obj_img -- np array, showing closest object only (having same dim as input)
    """
    # sharpen the intensity by * 30
    obj_img = (inst_idx == target_idx) * d1_img * 30
    return obj_img

