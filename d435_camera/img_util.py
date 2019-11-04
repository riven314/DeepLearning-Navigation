"""
AUTHOR: Alex Lau

SUMMARY
utility function for processing images, such as:
1. image resize
2. image crop
3. loading image from .npy file

REFERENCE

"""
import numpy as np
import cv2

def load_img(r_path):
    x = np.load(r_path)
    #h, w, c = x.shape
    print('Image Loaded = {}'.format(r_path))
    print('Size (h, w, c) = {}'.format(x.shape))
    return x


def reshape_img(img, expect_h = 300):
    """
    resize to ~300 in one size, and then do central cropping.
    crop_img is a copy from img, no change in place

    input:
        img -- np array, (h, w, 3)
    return 
        new_img -- np array, (expect, expect, 3)
    """
    h, w = img.shape[:2]
    aspect = float(w) / float(h)
    expect_w = round(expect_h * aspect)
    # cv2.resize(img, (width, height))
    resized_img = cv2.resize(img, (expect_w, expect_h))
    crop_start = round(expect_h * (aspect - 1) / 2)
    # central cropping
    crop_img = resized_img[0:expect_h, crop_start:crop_start + expect_h]
    return crop_img


if __name__ == '__main__':
    pass
