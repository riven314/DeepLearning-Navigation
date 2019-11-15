"""
grouping labels 
e.g. 4 (floor)     <- 7(road), 14(ground), 30(field), 53(path), 55(runway), 12(sidewalk)
edit both names and colors
"""

import numpy as np

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

    THEN, rename the grouped labels
    """
    colors[76] = np.array([6, 230, 230], dtype = np.uint8)
    names[77] = 'objects'
    for i in [1,9,19,33,43,44,145,11,9,101]:
        names[i] = 'wall'
    for i in [4,7,14,30,53,55,12,54]:
        names[i] = 'floor'
    for i in [5,18]:
        names[i] = 'plant'
    for i in [8,14,16,20,25,34,32,76]:
        names[i] = 'furniture'
    return colors, names