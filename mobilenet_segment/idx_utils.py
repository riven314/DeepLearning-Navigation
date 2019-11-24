"""
grouping labels 
e.g. 4 (floor)     <- 7(road), 14(ground), 30(field), 53(path), 55(runway), 12(sidewalk)
edit both names and colors

DICTIONARY:
0  wall      [120, 120, 120] --> 0
3  floor     [80, 50, 50]    --> 1
4  plant     [4, 200, 3]     --> 2
5  ceiling   [120, 120, 80]  --> 3
7  furniture [204, 5, 255]   --> 4
12 person    [150, 5, 61]    --> 5
14 door      [8, 255, 51]    --> 6
76 objects   [6, 230, 230]   --> 7
"""

import numpy as np

def create_idx_group(n = 150):
    """
    all are index
    e.g. all indexes: [8,18,32,42,43,144,10,8,100] mapping to wall_0 (index 0)
    the rest map to index 76 (boat, names[77])
    """
    idx_map = np.array([7 for i in range(n)])
    # group wall
    wall_0 = np.array([0,8,18,32,42,43,144,10,8,100])
    idx_map[wall_0] = 0
    # group  floor
    floor_3 = np.array([3,6,13,29,52,54,11,53])
    idx_map[floor_3] = 1
    # group tree
    tree_4 = np.array([4,17])
    idx_map[tree_4] = 2
    # group furniture
    furniture_7 = np.array([7,13,15,19,24,33,31,75])
    idx_map[furniture_7] = 4
    # group door
    door_14 = np.array([14,58])
    idx_map[door_14] = 6
    # idx:5 (ceiling) , idx:12 (person)
    idx_map[5] = 3
    idx_map[12] = 5
    return idx_map


def edit_colors_names_group(colors, names):
    """
    idx map to colors and its names. names index is 1 increment up
    based on new indexes (only 0 - 7)
    """
    # wall
    colors[0] = np.array([120, 120, 120], dtype = np.uint8)
    names[1] = 'wall'
    # floor
    colors[1] = np.array([80, 50, 50], dtype = np.uint8)
    names[2] = 'floor'
    # plant
    colors[2] = np.array([4, 200, 3], dtype = np.uint8)
    names[3] = 'plant'
    # ceiling
    colors[3] = np.array([120, 120, 80], dtype = np.uint8)
    names[4] = 'ceiling'
    # furniture
    colors[4] = np.array([204, 5, 255], dtype = np.uint8)
    names[5] = 'furniture'
    # person
    colors[5] = np.array([150, 5, 61], dtype = np.uint8)
    names[6] = 'person'
    # door
    colors[6] = np.array([8, 255, 51], dtype = np.uint8)
    names[7] = 'door'
    # objects
    colors[7] = np.array([6, 230, 230], dtype = np.uint8)
    names[8] = 'objects'
    return colors, names