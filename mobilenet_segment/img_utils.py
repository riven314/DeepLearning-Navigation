"""
way to optimise:
1. list copy size = ensemble size (easy)
2. replace PIL by cv2 for resizing (easy)
3. multiprocessing on image processing (require, GPU, CPU shared memory)
4. multiprocessing on model feed (require, GPU shared memory)
5. visualize_result color encoder (easy, CPU)
6. change imgSizes list scale (easy)

REFERENCE:
1. different way of multiprocessing: https://stackoverflow.com/questions/56094689/overhead-of-python-multiprocessing-initialization-is-worse-than-benefits
"""
import os
import sys
import time

# multiprocessing
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2

from profiler import profile

def ImageLoad(data, width, height, ensemble_n, is_silent):
    """
    read the image data and resize it.
    this is for performance comparison
    
    Input:
    data: the image data which is numpy array with shape: (3, W, H)
    
    output:
    A dict of image, which has two keys: 'img_ori' and 'img_data'
    the value of the key 'img_ori' means the original numpy array
    the value of the key 'img_data' is the list of five resize images 
    """
    normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )
    # transfrom the numpy array to image
    # PIL.Image.fromarray is slow!
    img = Image.fromarray(data)
    #change the image size
    img=img.resize((width,height), resample = Image.BILINEAR)
    ori_width, ori_height = img.size

    device = torch.device("cuda", 0)
    p=8 #padding_constant value
    imgSizes = [300,375,450,525,600] 
    #imgSizes = [100, 175, 240, 525, 600] 
    imgMaxSize = 1000
    #  above three value are got from cfg file  
    
    img_resized_list=[]
    for this_short_size in imgSizes:
    #for i in range(ensemble_n):
        #this_short_size = imgSizes[i]
        #calculate target height and width
        scale=min(this_short_size / float(min(ori_height, ori_width)),
                  imgMaxSize / float(max(ori_height,ori_width)))
        target_height, target_width = int(ori_height*scale), int(ori_width*scale)
        
        #to avoid rounding in network
        # Round x to the nearest multiple of p and x' >= x
        target_width = ((target_width-1)//p+1)*p
        target_height = ((target_height-1)//p+1)*p
        
        #resize images
        img_resize = img.resize((target_width, target_height), resample = Image.BILINEAR)
        
        #image transform, to torch float tensor 3xHxW
        img_resized = np.float32(img_resize)/255
        img_resized = img_resized.transpose((2,0,1))
        # send to GPU earlier leads to speed up + more CPU efficient
        img_resized = normalize(torch.from_numpy(img_resized.copy()).to(device))
        # it is in CPU mode!
        #img_resized=normalize(torch.from_numpy(img_resized.copy()))

        img_resized = torch.unsqueeze(img_resized,0)
        img_resized_list.append(img_resized)

    output=dict()
    output['img_ori'] = np.array(img)
    if not is_silent:
        print('img size', np.array(img).shape)
    output['img_data'] = [x.contiguous() for x in img_resized_list]
    return output


def rescale_img(args):
    """
    copy a image and resize with target scale 

    input:
        ori_height -- original height of image
        ori_width -- original width of image
        normalize -- torch.transforms
        img -- np array, (H, W, 3)
        target_scale -- int, e.g. any one of [300, 375, 450, 525, 600] 
        img_max_size -- max image size after resizing
        p -- int, padding value
        device -- CPU or GPU
    output:
        img_resized -- torch tensor after resizing
    """
    ori_height = args[0]
    ori_width = args[1]
    normalize = args[2]
    img = args[3]
    target_scale = args[4]
    img_max_size = 1000
    p = 8
    scale = min(target_scale / float(min(ori_height, ori_width)),
                img_max_size / float(max(ori_height, ori_width)))
    target_height, target_width = int(ori_height * scale), int(ori_width * scale)
    
    #to avoid rounding in network
    # Round x to the nearest multiple of p and x' >= x
    target_width = ((target_width - 1) // p + 1) * p
    target_height = ((target_height - 1) // p + 1) * p
    
    #[cv2 approach] resizing image
    img_resize = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
    
    #image transform, to torch float tensor 3xHxW
    img_resized = np.float32(img_resize) / 255
    img_resized = img_resized.transpose((2,0,1))
    return img_resized


def ImageLoad_cv2_parallize(data, width, height, ensemble_n, is_silent):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    #[cv2 approach]
    img = data
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    ori_height, ori_width, _ = img.shape

    imgSizes = [300, 375, 450, 525, 600]
    imgSizes = imgSizes[:ensemble_n]
    args_ls = [[ori_height, ori_width, normalize, img.copy(), scale] for scale in imgSizes]
    #  above three value are got from cfg file  
    
    pool = Pool(processes = ensemble_n)
    img_resized_list = pool.map(rescale_img, args_ls)
    pool.close()
    return img_resized_list


def ImageLoad_cv2(data, width, height, ensemble_n, is_silent):
    """
    read the image data and resize it.
    
    Input:
    data: the image data which is numpy array with shape: (H, W, 3)
    
    output:
    A dict of image, which has two keys: 'img_ori' and 'img_data'
    the value of the key 'img_ori' means the original numpy array
    the value of the key 'img_data' is the list of five resize images 
    """
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    #change the image size
    #[cv2 approach]
    img = data
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    ori_height, ori_width, _ = img.shape

    p = 8 #padding_constant value
    imgSizes = [300, 375, 450, 525, 600] 
    #imgSizes = [375, 450, 525, 465, 540] #(0.087-0.095s per process + predict )
    imgMaxSize = 1000
    #  above three value are got from cfg file  
    
    img_resized_list = []
    if ensemble_n == 1:
        print('one img')
        img_resized = np.float32(img)/255
        img_resized = img_resized.transpose((2,0,1))
        img_resized=normalize(torch.from_numpy(img_resized).cuda())
        img_resized=torch.unsqueeze(img_resized,0)
        img_resized_list.append(img_resized)
        output=dict()
        output['img_ori'] = np.array(img)
        if not is_silent:
            print('img size', np.array(img).shape)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        return output

    for i in range(ensemble_n):
        this_short_size = imgSizes[i]
        #calculate target height and width
        scale=min(this_short_size / float(min(ori_height, ori_width)),
                  imgMaxSize / float(max(ori_height,ori_width)))
        target_height, target_width = int(ori_height*scale), int(ori_width*scale)
        
        #to avoid rounding in network
        # Round x to the nearest multiple of p and x' >= x
        target_width = ((target_width-1)//p+1)*p
        target_height = ((target_height-1)//p+1)*p
        
        #[cv2 approach] resizing image
        img_resize = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
        
        #image transform, to torch float tensor 3xHxW
        img_resized=np.float32(img_resize)/255
        img_resized=img_resized.transpose((2,0,1))
        # send to GPU earlier leads to speed up + more CPU efficient
        img_resized=normalize(torch.from_numpy(img_resized).cuda())
        # it is in CPU mode!
        #img_resized=normalize(torch.from_numpy(img_resized.copy()))

        img_resized=torch.unsqueeze(img_resized,0)
        img_resized_list.append(img_resized)

    output=dict()
    output['img_ori'] = np.array(img)
    if not is_silent:
        print('img size', np.array(img).shape)
    output['img_data'] = [x.contiguous() for x in img_resized_list]
    return output


if __name__ == '__main__':
    WIDTH = 484
    HEIGHT = 240
    ENSEMBLE_N = 3
    data = np.load(os.path.join('test_set', 'cls1_rgb.npy'))
    data = data[:, :, ::-1]
    for i in range(5):
        # time ImageLoad
        torch.cuda.synchronize()
        start = time.time()
        out = ImageLoad(data, WIDTH, HEIGHT, ENSEMBLE_N, True)
        torch.cuda.synchronize()
        end = time.time()
        print('ImageLoad runtime: {}s'.format(end - start))
        
        # time cv2
        torch.cuda.synchronize()
        start = time.time()
        out_cv2 = ImageLoad_cv2(data, WIDTH, HEIGHT, ENSEMBLE_N, True)
        torch.cuda.synchronize()
        end = time.time()
        print('ImageLoad_cv2 runtime: {}s'.format(end - start))
