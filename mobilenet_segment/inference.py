"""
guide on speeding up and unit testing:
1. if PIL is used, truncate image list (5 copy -> 3 copy)
2. use cv2.resize instead (may affect segmentation result)
3. add more unit test cases on segmentation result if cv2.resize is used
4. try to parallelize resize
5. double check if upsampling is down for small image
"""
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
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
import time
from profiler import profile
from torchvision import transforms
import cv2

assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

def ImageLoad(data, width, height, is_silent):
    """
    read the image data and resize it.
    
    Input:
    data: the image data which is numpy array with shape: (3, W, H)
    
    output:
    A dict of image, which has two keys: 'img_ori' and 'img_data'
    the value of the key 'img_ori' means the original numpy array
    the value of the key 'img_data' is the list of five resize images 
    """
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    # transfrom the numpy array to image
    # PIL.Image.fromarray is slow!
    img = Image.fromarray(data)
    #change the image size
    img=img.resize((width,height), resample = Image.BILINEAR)
    ori_width, ori_height = img.size
    #[cv2 approach]
    #img = data
    #img = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    #ori_height, ori_width, _ = img.shape

    device = torch.device("cuda", 0)
    p=8 #padding_constant value
    imgSizes=[300,375,450,525,600] # [300,375,450,525,600]
    imgMaxSize=1000
    #  above three value are got from cfg file  
    
    img_resized_list=[]
    for this_short_size in imgSizes:
        #calculate target height and width
        scale=min(this_short_size/float(min(ori_height, ori_width)),
                  imgMaxSize/float(max(ori_height,ori_width)))
        target_height, target_width=int(ori_height*scale), int(ori_width*scale)
        #print('target_height = {}, target_width = {}'.format(target_height, target_width))
        
        #to avoid rounding in network
        # Round x to the nearest multiple of p and x' >= x
        target_width=((target_width-1)//p+1)*p
        target_height=((target_height-1)//p+1)*p
        
        #resize images
        img_resize=img.resize((target_width, target_height), resample = Image.BILINEAR)
        #[cv2 approach]
        #img_resize = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
        
        #image transform, to torch float tensor 3xHxW
        img_resized=np.float32(img_resize)/255
        img_resized=img_resized.transpose((2,0,1))
        # send to GPU earlier leads to speed up + more CPU efficient
        img_resized=normalize(torch.from_numpy(img_resized.copy()).to(device))
        # it is in CPU mode!
        #img_resized=normalize(torch.from_numpy(img_resized.copy()))

        img_resized=torch.unsqueeze(img_resized,0)
        img_resized_list.append(img_resized)

    output=dict()
    output['img_ori']=np.array(img)
    if not is_silent:
        print('img size',np.array(img).shape)
    output['img_data']=[x.contiguous() for x in img_resized_list]
    return output


def visualize_result(pred, colors, names, is_silent):
    """
    input: the predictions (np.array), shape is (height, width)
    
    output: colorize prediction whose shape is (height, width, 3) and print the predictions ratio in descending order
    """
    # print predictions result in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8) # this spends time
    if is_silent:
        return pred_color
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
    return pred_color
    
    
    
def predict(model, image_load, resizeNum, is_silent, gpu=0):
    """
    input:
    model: model
    image_load: A dict of image, which has two keys: 'img_ori' and 'img_data'
    the value of the key 'img_ori' means the original numpy array
    the value of the key 'img_data' is the list of five resize images 
    
    output:
    the mean predictions of the resize image list: 'img_data' 
    """
    starttime = time.time()
    segSize = (image_load['img_ori'].shape[0],
               image_load['img_ori'].shape[1])
    #print('segSize',segSize)
    img_resized_list = image_load['img_data']
    with torch.no_grad():
        scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1], device=torch.device("cuda", gpu))
        
        for img in img_resized_list[:resizeNum]:
            feed_dict = image_load.copy()
            feed_dict['img_data']=img
            del feed_dict['img_ori']
            feed_dict=async_copy_to(feed_dict, gpu)
            
            # forward pass
            pred_tmp = model(feed_dict, segSize = segSize) #shape of pred_temp is (1, 150, height, width)
            scores = scores + pred_tmp / resizeNum
    endtime = time.time()
    if not is_silent:
        print('model inference time: {}s' .format(endtime-starttime))
    return scores


def process_predict(scores, colors, names, is_silent):
    """
    input:
    the predictions of model
    
    output:
    the colorize predictions
    """
    _, pred = torch.max(scores, dim=1)
    pred = as_numpy(pred.squeeze(0).cpu()) # shape of pred is (height, width)
    #The predictions for infering distance
    seg = np.moveaxis(pred, 0, -1)
    pred_color = visualize_result(pred, colors, names, is_silent)
    return seg, pred_color


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

# the final function we use 
def webcam_predict(data, cfg_path, root, colors, names, width, height,resizeNum):
    Image = ImageLoad(data, width, height, is_silent = False)
    model = setup_model(cfg_path, root, gpu=0)
    model.eval()
    predictions = predict(model,Image,resizeNum,gpu=0, is_silent = False)
    seg,pred_color = process_predict(predictions, colors, names, is_silent = False)
    return seg,pred_color


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
    WIDTH = 484 # 484
    HEIGHT = 240
    RESIZE_N = 3
    colors = loadmat('data/color150.mat')['colors']
    root = ''
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]        
        
    #take cls.npy as an example
    data = np.load('test_set/cls1_rgb.npy')
    data = data[:, :, ::-1]
    #plt.imshow(data)
    #plt.show()
    cfg_path = "config/ade20k-mobilenetv2dilated-c1_deepsup.yaml"
    #cfg_path="config/ade20k-resnet18dilated-ppm_deepsup.yaml"
    Image = ImageLoad(data, WIDTH, HEIGHT, is_silent = False)
    model = setup_model(cfg_path, root, gpu=0)
    model.eval()
    for i in range(10):
        start = time.time()
        predictions = predict(model, Image, RESIZE_N, gpu = 0, is_silent = True)
        end = time.time()
        print('process + prediction: {}s'.format(end - start))
        start = time.time()
        seg, pred_color = process_predict(predictions, colors, names, is_silent = True)
        end = time.time()
        print('visualize = {} s'.format(end - start))
    plt.imshow(pred_color)
    plt.show()
    #np.save('test_result.npy',pred_color) 
