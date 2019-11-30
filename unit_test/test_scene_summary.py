import os
import sys
import time
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scene_summary import get_names, get_small_portion, scene_summarize, create_grid

PRED_IDX_PATH = os.path.join(os.getcwd(), '..', 'test_cases', 'test2_pred_idx.png')
PRED_RGB_PATH = os.path.join(os.getcwd(), '..', 'test_cases', 'test2_pred_rgb.jpg')

# initialize
pred_idx = cv2.imread(PRED_IDX_PATH, cv2.IMREAD_GRAYSCALE)
pred_rgb = cv2.imread(PRED_RGB_PATH)
pred_rgb = pred_rgb[:, :, ::-1]
h, w = pred_idx.shape
names = get_names()
t = get_small_portion(h, w)
mat = create_grid(h, w)
for i in range(5):
    start = time.time()
    grid_summary = scene_summarize(pred_idx, mat, names, threshold = 900)
    end = time.time()
    print('runtime: {} s'.format(end - start))
print(grid_summary)

# visualize the result
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
ax1.imshow(pred_rgb)
ax2.imshow(mat)
plt.show()