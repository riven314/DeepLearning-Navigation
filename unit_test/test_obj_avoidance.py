import os
import sys
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from obj_avoidance import run_avoidance

# for the reference
label_dict = {1: 'wall', 2: 'floor', 3: 'plant', 4: 'ceiling', 5: 'furniture', 6: 'person', 7: 'door', 8: 'objects'}

# read in image
D1_IMG_PATH = os.path.join(os.getcwd(), '..', 'test_cases', 'test_obj_avoid_resize_d1.png')
SEG_IDX_PATH = os.path.join(os.getcwd(), '..', 'test_cases', 'test_obj_avoid_pred_idx.png')
d1_img = cv2.imread(D1_IMG_PATH, cv2.IMREAD_GRAYSCALE)
seg_idx = cv2.imread(SEG_IDX_PATH, cv2.IMREAD_GRAYSCALE)

obj_tup, obj_img = run_avoidance(d1_img, seg_idx, depth_threshold = 8, visible_width = 90)
print('obj_tup = {}'.format(obj_tup))

rgb_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2RGB)
plt.imshow(obj_img)
plt.show()
plt.imshow(rgb_img)
plt.show()

