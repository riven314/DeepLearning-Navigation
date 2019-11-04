"""
LOG
[15/10/2019]
- to do: add new window for model prediction
"""
import os
import sys
import time

import numpy as np

import pyrealsense2 as rs
import cv2

PATH = os.path.join(os.getcwd(), 'segmentation')
sys.path.append(PATH)
from segmentation.predict import model_predict, setup_model
from img_stream import warmup_camera, filter_depth

color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.286230, 0.291129]

def stream_caemra_wsegment(config, frame_limit, model, class_encoding, is_process_depth = False, is_align = True):
    """
    input:
        config -- rs.config class instance
        frame_limit -- int, number of frames to be print (if None, then endless stream)
        is_process_depth -- bool, whether apply filters on the depth frames
        is_align -- bool, whether align the viewpoint of RGB image and Depth image
    """
    global color_mean
    global color_std
    pipeline = warmup_camera(config)
    frame_cnt = 0
    if is_align:
        align = rs.align(rs.stream.color)
    while True:
        # retrieve rgb and depth frames
        if frame_cnt == frame_limit:
            break
        start = time.time()
        # frame waiting time is a bottleneck, lower resolution can improve this
        frames = pipeline.wait_for_frames() 
        frame_cnt += 1
        # RGBD alignment with respect to RGB image
        if is_align: 
            frames = align.process(frames)
        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not rgb_frame or not depth_frame:
            print('current frame corrupted, wait for next...')
            continue
        # massage two frames for display
        color_image = np.asanyarray(rgb_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # feed to model
        model_out = model_predict(color_image, depth_image, 
                                  color_mean, color_std,
                                  model, class_encoding,
                                  is_cuda = True)
        if is_process_depth:
            depth_frame = filter_depth(depth_frame)
        colorizer = rs.colorizer() # colorizer looks nice
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        #depth_image = np.asanyarray(depth_frame.get_data())
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), 
        #                                   cv2.COLORMAP_JET)
        display_images = np.concatenate((color_image, depth_colormap), axis=1)
        t = time.time() - start
        # display on windows
        print('Frame No: {}'.format(frame_cnt))
        try:
            print('FPS = {}'.format(1. / t))
        except:
            print('FPS = NaN')
        # window for RGBD input
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL) # may needa change autosize
        cv2.imshow('Align Example', display_images)
        # window for segmentation result
        if model_out is not None:
            cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
            cv2.imshow('Prediction', model_out)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    pipeline.stop()
    cv2.destroyAllWindows()
    print('Streaming Stop!')
    return color_image, depth_colormap

if __name__ == '__main__':
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    model, class_encoding = setup_model(root = 'segmentation')
    stream_caemra_wsegment(config, 20, model, class_encoding)    