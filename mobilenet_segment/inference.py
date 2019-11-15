# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode_numpy, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
import time
from profiler import profile
from torchvision import transforms
import cv2
from img_utils import ImageLoad_cv2
from idx_utils import create_idx_group, edit_colors_names_group

assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'


def visualize_result(pred, colors, names, is_silent):
    """
    input: the predictions (np.array), shape is (height, width)
    
    output: colorize prediction whose shape is (height, width, 3) and print the predictions ratio in descending order
    """
    # print predictions result in descending order
    pred = np.int32(pred)
    # colorize prediction
    pred_color = colorEncode_numpy(pred, colors)
    if is_silent:
        return pred_color
    # compute only when verbose = True (it costs 2 ms)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}% (idx: {})".format(name, ratio, uniques[idx]))
    return pred_color
    
  
def predict(model, img_load, resizeNum, is_silent, gpu=0):
    """
    input:
    model: model
    img_load: A dict of image, which has two keys: 'img_ori' and 'img_data'
    the value of the key 'img_ori' means the original numpy array
    the value of the key 'img_data' is the list of five resize images 
    
    output:
    the mean predictions of the resize image list: 'img_data' 
    """
    starttime = time.time()
    segSize = (img_load['img_ori'].shape[0],
               img_load['img_ori'].shape[1])
    #print('segSize',segSize)
    img_resized_list = img_load['img_data']
    with torch.no_grad():
        scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1], device=torch.device("cuda", gpu))
        
        for img in img_resized_list:
            feed_dict = img_load.copy()
            feed_dict['img_data']=img
            del feed_dict['img_ori']
            #feed_dict = {'img_data': img}
            feed_dict=async_copy_to(feed_dict, gpu)
            
            # forward pass
            pred_tmp = model(feed_dict, segSize = segSize) #shape of pred_temp is (1, 150, height, width)
            scores = scores + pred_tmp / resizeNum
    endtime = time.time()
    if not is_silent:
        print('model inference time: {}s' .format(endtime-starttime))
    return scores


def process_predict(scores, colors, names, idx_map, is_silent):
    """
    input:
    the predictions of model
    
    output:
    the colorize predictions
    """
    _, pred = torch.max(scores, dim=1)
    pred = as_numpy(pred.squeeze(0).cpu()) # shape of pred is (height, width)
    # grouping label index
    pred = idx_map[pred]
    pred_color = visualize_result(pred, colors, names, is_silent)
    return pred_color


# model part
def setup_model(cfg_path, root, gpu=0):
    cfg.merge_from_file(cfg_path)
    
    # cfg.freeze()
    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(cfg_path))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        root, cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        root, cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
    
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    
    return segmentation_module


def InferDist(depth, seg, x,y,r):
    """
    input:
    depth: 1d depth data which should be numpy array with (width, height)
    seg: segmentation result which should be numpy array with (width, height)
    
    output:
    the distance of object(x,y) and the distance is calcuated by taking the mean of distances of points near the (x,y) with same class. 
    """
    assert depth.shape==seg.shape, "The sizes of Depth image and segmentation result are different! "
    assert 0<=x<=seg.shape[0] and 0<=y<=seg.shape[1], "The input of (x,y) is out of range!"
    cls=seg[x][y]
    x1=max(0,x-r)
    y1=max(0,y-r)
    x2=min(x+r,seg.shape[0])
    y2=min(y+r,seg.shape[1])
    print('The number of same class in the range: {}.' .format(np.sum(seg[x1:x2, y1:y2]==cls)))
    depth_range=depth[x1:x2, y1:y2]
    seg_range=seg[x1:x2, y1:y2]
    distance=np.mean(depth_range[seg_range==cls])
    return distance


if __name__ == '__main__':
    #Define the color dict
    import matplotlib.pyplot as plt
    WIDTH = 484
    HEIGHT = 240
    RESIZE_N = 2
    IS_SILENT = True
    colors = loadmat('data/color150.mat')['colors']
    root = ''
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    idx_map = create_idx_group()
    colors, names = edit_colors_names_group(colors, names)       
        
    #take cls.npy as an example
    data = np.load(os.path.join('test_set', 'cls1_rgb.npy'))
    data = data[:, :, ::-1]
    cfg_path = os.path.join('config', 'ade20k-mobilenetv2dilated-c1_deepsup.yaml')
    #cfg_path="config/ade20k-resnet18dilated-ppm_deepsup.yaml"
    model = setup_model(cfg_path, root, gpu=0)
    model.eval()
    for i in range(5):
        torch.cuda.synchronize()
        start = time.time()
        img = ImageLoad_cv2(data, WIDTH, HEIGHT, RESIZE_N, is_silent = IS_SILENT)
        predictions = predict(model, img, RESIZE_N, gpu=0, is_silent = IS_SILENT)
        torch.cuda.synchronize()
        end = time.time()
        print('process + predict: {}s'.format(end - start))
        torch.cuda.synchronize()
        start = time.time()
        pred_color = process_predict(predictions, colors, names, idx_map, is_silent = IS_SILENT)
        torch.cuda.synchronize()
        end = time.time()
        print('visualize: {}s'.format(end - start))
    plt.imshow(pred_color)
    plt.show()
