"""
AUTHOR: Alex Lau

SUMMARY
do all the heavy-lifting jobs for real-time streaming and image processing

REFERENCE
1. depth postprocessing parameters: 
    - discussion 1: https://github.com/IntelRealSense/librealsense/issues/2088
    - discussion 2: https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md
    - jupyter demo: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
2. depth image is halved with decimation
    - https://github.com/IntelRealSense/librealsense/issues/1284

LOG
[08/10/2019]
- currently, RGB image and D image are not aligned in viewpoint

"""
import os
import sys
import time

import pyrealsense2 as rs
import numpy as np
import cv2

def filter_depth(frame):
    """
    some filtering processes will greatly decrease FPS (e.g. spatial)
    decimation process will reduce image size exponentially

    reference: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
    """
    # setup filter processes
    #decimation = rs.decimation_filter() # decimation will cut frame size (exponentially)
    #depth_to_disparity = rs.disparity_transform(True)
    #spatial = rs.spatial_filter() # spatial operation is quite slow
    #temporal = rs.temporal_filter()
    #disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter()
    # apply the filters
    #frame = depth_to_disparity.process(frame)
    #frame = spatial.process(frame)
    #frame = temporal.process(frame)
    #frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)
    return frame


def warmup_camera(config, n_trial = 20):
    """
    input:
        config -- rs.config class instance
        n_trial -- number of frames for depth auto-adjustment
    """
    pipeline = rs.pipeline()
    pipeline.start(config)
    for x in range(n_trial):
        pipeline.wait_for_frames()
    print('camera warmup complete!')
    return pipeline


def process_frame(rgb_frame, depth_frame):
    """
    assume rgb_frame or depth_frame is not None, process them into np array

    output:
        rgb_img -- np array, rgb image 
        depth_img -- np array, 1 channel depth image
        depth_colormap -- np array, 3 channel depth colormap
    """
    rgb_img = np.asanyarray(rgb_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    # get depth colormap
    colorizer = rs.colorizer() # colorizer looks nice
    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    return rgb_img, depth_img, depth_colormap


def stream_camera(camera, frame_limit, is_process_depth = False, is_align = True):
    """
    input:
        camera -- RGBDhandler
        frame_limit -- int, number of frames to be print (if None, then endless stream)
        is_process_depth -- bool, whether apply filters on the depth frames
        is_align -- bool, whether align the viewpoint of RGB image and Depth image
    """
    #pipeline = warmup_camera(config)
    frame_cnt = 0
    if is_align:
        align = rs.align(rs.stream.color)
    while True:
        # retrieve rgb and depth frames
        if frame_cnt == frame_limit:
            break
        start = time.time()
        # frame waiting time is a bottleneck, lower resolution can improve this
        frames = camera.pipeline.wait_for_frames() 
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
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL) # may needa change autosize
        cv2.imshow('Align Example', display_images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    camera.pipeline.stop()
    cv2.destroyAllWindows()
    print('Streaming Stop!')
    return color_image, depth_image, depth_colormap


if __name__ == '__main__':
    pass