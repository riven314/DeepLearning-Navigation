"""
current pytorch performance: 0.02 - 0.032 s

"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torchvision import transforms

# (N, C, H, W)
#t = torch.randint(0, 255, size = (1, 3, 720, 1280), dtype = torch.uint8)

def set_resize_layers(p_ls):
    resize_m_ls = []
    for p in p_ls:
        m = nn.Upsample(scale_factor = p, mode = 'bilinear', align_corners = False)
        resize_m_ls.append(m)
    return resize_m_ls


def np_to_uint_tensor(np_data):
    """
    permute and put numpy to cuda tensor, convert to float 0-1
    
    input:
        np_data -- np array, (H, W, C), uint8
    output:
        t -- torch tensor, (C, H, W), float32
    """
    np_data = np.float32(np_data) / 255
    np_data = np_data.transpose(2, 0, 1)
    t = torch.from_numpy(np_data).cuda()
    return t


def ImageLoad_torch(data, ensemble_n, resize_layers, is_silent):
    """
    input:
        data -- np array (H, W, 3)
        p -- rescale factor. e.g. p=0.4: (720, 1280) --> (288, 512)
    """
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
    img_t = np_to_uint_tensor(data.copy())
    # normalize and then resize
    img_t = normalize(img_t)
    img_t = img_t.unsqueeze(0)
    img_t = resize_layers[0](img_t)
    _, _, ori_height, ori_width = img_t.shape
    #imgSizes = [300, 375, 450, 525, 600]
    #imgMaxSize = 1000
    img_resized_list = []

    for i in range(ensemble_n):
        m = resize_layers[i + 1]
        img_t_resized = m(img_t)
        img_resized_list.append(img_t_resized)

    output=dict()
    output['img_ori'] = data
    if not is_silent:
        print('img size', img_t.shape)
    output['img_data'] = [x.contiguous() for x in img_resized_list]
    return output


####################### TESTING ###########################
def test_resize_layers():
    p_ls = [0.4, 1.041667, 1.302083, 1.5625, 1.82292]
    resize_layers = set_resize_layers(p_ls)
    return resize_layers


def test_np_to_t():
    x = np.random.randint(0, 255, (720, 1280, 3), dtype = np.uint8)
    t = np_to_uint_tensor(x)
    return t


if __name__ == '__main__':
    #t = test_np_to_t()
    resize_layers = test_resize_layers()
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        img = np.random.randint(0, 255, (720, 1280, 3), dtype = np.uint8)
        out = ImageLoad_torch(img, 3, resize_layers, is_silent = True)
        torch.cuda.synchronize()
        end = time.time()
        print('torch resize runtime: {}s'.format(end - start))

