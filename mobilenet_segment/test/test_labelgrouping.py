import os
import sys
import time
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import csv
import numpy as np
from scipy.io import loadmat

def fetch_raw_colors_names():
    # GET COLOR ENCODING AND ITS INDEX MAPPING
    colors = loadmat('../data/color150.mat')['colors']
    root = '..'
    names = {}
    with open('../data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    return colors, names


def create_idx_group(n = 150):
    """
    all are index
    e.g. all indexes: [8,18,32,42,43,144,10,8,100] mapping to wall_0 (index 0)
    the rest map to index 76 (boat, names[77])
    """
    idx_map = np.array([76 for i in range(n)])
    # group wall
    wall_0 = np.array([0,8,18,32,42,43,144,10,8,100])
    idx_map[wall_0] = 0
    # group  floor
    floor_3 = np.array([3,6,13,29,52,54,11,53])
    idx_map[floor_3] = 3
    # group tree
    tree_4 = np.array([4,17])
    idx_map[tree_4] = 4
    # group furniture
    furniture_7 = np.array([7,13,15,19,24,33,31,75])
    idx_map[furniture_7] = 7
    # group door
    door_14 = np.array([14,58])
    idx_map[door_14] = 14
    # idx:5 (ceiling) , idx:12 (person)
    idx_map[5] = 5
    idx_map[12] = 12
    return idx_map


def edit_colors_names_group(colors, names):
    """
    colors[76] = [6, 230, 230]
    names[77] = 'objects'
    """
    colors[76] = np.array([6, 230, 230], dtype = np.uint8)
    names[77] = 'objects'
    for i in [1,9,15,19,33,43,44,145,11,9,101]:
        names[i] = 'wall'
    for i in [4,7,14,30,53,55,12, 54]:
        names[i] = 'floor'
    for i in [5,18]:
        names[i] = 'plant'
    for i in [8,14,16,20,25,34,32,76]:
        names[i] = 'furniture'
    return colors, names


def group_colors_names(colors, names):
    """
    from Maggie
    """
    #1 (wall) <- 9(window), 33(fence), 43(pillar), 44(sign board), 145(bullertin board), 11(cabinet), 9(windowpane)，101(poster)
    colors[[8,18,32,42,43,144,10,8,100]] = colors[0]
    #4 (floor)     <- 7(road), 14(ground), 30(field), 53(path), 55(runway), 12(sidewalk)
    colors[[6,13,29,52,54,53,11]] = colors[3]
    #5 (tree)      <- 18(plant)
    colors[17] = colors[4]
    #8 (furniture) <- 8(bed), 14(sofa), 16(table), 20(chair), 25(shelf), 34(desk), 32(seat), 76(swivel)
    colors[[7,13,15,19,24,33,31,75]] = colors[7]
    #15 (door) <- 59 (screen)
    colors[58] = colors[14]  #door/screen
    #6 (ceiling) , 13 (person)
    for i in range(1,len(colors)):
        if i not in [1,9,15,19,33,43,44,145,11,9,101,4,7,14,30,53,55,12,18,5,8,14,16,20,25,34,32,76,54,6,13,59]:
            names[i] = 'objects'
            colors[i-1] = [6, 230, 230]  #building
    for i in [1,9,15,19,33,43,44,145,11,9,101]:
        names[i] = 'wall'
    for i in [4,7,14,30,53,55,12, 54]:
        names[i] = 'floor'
    for i in [5,18]:
        names[i] = 'plant'
    for i in [8,14,16,20,25,34,32,76]:
        names[i] = 'furniture'
    return colors, names


if __name__ == '__main__':
    old_colors, old_names = fetch_raw_colors_names()
    idx_map = create_idx_group()
    new_colors, new_names = edit_colors_names_group(old_colors.copy(), old_names.copy())

