"""
PROCEDURES
1. create a grid with size = prediction map size
2. merge pred idx with grid
3. uniques counts
4. combine with name map to do a summary
5. test on 4 different test cases

REMARK
1. time the speed of scene_summarize()

REFERENCE
1. making dict comprehension: https://cmdlinetips.com/2018/01/5-examples-using-dict-comprehension/
"""
import os
import sys
import time
from collections import defaultdict

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt


# get test case
ROOT = os.path.join(os.getcwd(), '..', 'test_cases')
PRED_IDX_F = 'test{}_pred_idx.png'
PRED_RGB_F = 'test{}_pred_rgb.jpg'
test_dict = {i: [os.path.join(ROOT, PRED_IDX_F.format(i)), os.path.join(ROOT, PRED_RGB_F.format(i))] for i in range(1, 5)}


# get names
def get_names():
    names = {}
    names[1] = 'wall'
    names[2] = 'floor'
    names[3] = 'plant'
    names[4] = 'ceiling'
    names[5] = 'furniture'
    names[6] = 'person'
    names[7] = 'door'
    names[8] = 'objects'
    return names


def get_small_portion(h, w, grid = (2, 3)):
    """
    in a grid, how many pixels are consider small?
    """
    grid_r, grid_c = grid
    grid_h = int(h / grid_r)
    grid_w = int(w / grid_c)
    grid_n = grid_h * grid_w
    threshold = int(grid_n * 0.05)
    print('(h = {}, w= {}), grid = ({})'.format(h, w, grid))
    print('grid size = {}'.format(grid_n))
    print('threshold = {}'.format(threshold))
    return threshold


def create_grid(h, w, grid = (2, 3)):
    """
    create a numpy array with (h, w)
    divide into region_n grids, each have different value
    idx from left to right, top to dow

    input:
        grid -- tuple, (# row grid, # col grid)
    """
    mat = np.zeros(shape = (h, w), dtype = np.uint8)
    grid_r, grid_c = grid
    grid_h = int(h / grid_r)
    grid_w = int(w / grid_c)
    mapping = {1: 0, 2: 10, 3: 20, 11: 30, 12: 40, 13: 50}
    for i in range(grid_r):
        for j in range(grid_c):
            idx = 10*i + j + 1
            val = mapping[idx]
            # the end of row
            if i == grid_r - 1 and j == grid_c - 1:
                mat[i*grid_h :  , j*grid_w : ] = val
            elif i == grid_r - 1:
                mat[i*grid_h :  , j*grid_w : (j+1)*grid_w] = val
            # the end of column
            elif j == grid_c - 1:
                mat[i*grid_h : (i+1)*grid_h , j*grid_w : ] = val
            else:    
                mat[i*grid_h : (i+1)*grid_h , j*grid_w : (j+1)*grid_w] = val
    return mat


def filter_idxs(idxs, counts, threshold = 900):
    """
    kick out idxs if its count <= grid sizes * 0.05

    input:
        idxs, counts -- output from np.unique(. , return_counts = True)
    return:
        new_idxs
    """
    new_idxs = [i for i, c in zip(idxs, counts) if c > threshold]
    return new_idxs


def scene_summarize(pred_idx, mat, names, threshold = 900):
    new_pred_idx = pred_idx + mat
    idxs, counts = np.unique(new_pred_idx, return_counts = True)
    # kick out idxs with small counts
    new_idxs = filter_idxs(idxs, counts, threshold = threshold)
    grid_dict = grid_name_mapping(new_idxs, names)
    return grid_dict


def grid_name_mapping(new_idxs, names):
    """
    create a dict with key = grid index, value = list of objects in the grid

    input:
        new_idxs -- list of new indexes (with grid adding) after count filtering
        names -- dict of names mapping from index
    output:
        grid_dict: {grid_idx: names}
    """
    grid_dict = defaultdict(list)
    for idx in new_idxs:
        obj = names[idx%10 + 1]
        grid_dict[idx//10].append(obj)
    return grid_dict


def test_create_grid():
    """
    visualize the created numpy grid, check the dtype
    """
    h = 240
    w = 484
    grid = (2, 3)
    mat = create_grid(h, w, grid = (2, 3))
    n = grid[0] * grid[1]
    mat_h, mat_w = mat.shape
    idx, _ = np.unique(mat, return_counts = True)
    assert h == mat_h and w == mat_w, 'WRONG SHAPE'
    assert len(idx) == n, 'WRONG FILLING'
    plt.imshow(mat)
    plt.show()
    return mat


if __name__ == '__main__':
    h = 240
    w = 484
    names = get_names()
    t = get_small_portion(h, w)
    idx_f = test_dict[2][0]
    pred_idx = cv2.imread(idx_f, cv2.IMREAD_GRAYSCALE)
    # run this script in "test" dir
    mat = create_grid(h, w)
    for i in range(5):
        start = time.time()
        grid_summary = scene_summarize(pred_idx, mat, names, threshold = 900)
        end = time.time()
        print('runtime: {} s'.format(end - start))
    print(grid_summary)