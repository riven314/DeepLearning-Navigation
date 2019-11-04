"""
AUTHOR: Alex Lau

SUMMARY
Configure the RGBD camera handler before streaming, parameters include:
1. depth scale for depth accuracy
2. resolution
3. color format
4. FPS
5. minimum depth distance
6. postprocessing on depth
7. realtime streaming for sample tests

LOG
[06/10/2019]
- adjust depth accuracy
- check if GPU is connected
"""
import os
import sys
import time

import pyrealsense2 as rs
import numpy as np
import cv2

from img_stream import stream_camera

class RGBDhandler:
    def __init__(self, rgb_res, rgb_format, depth_res, depth_format, fps):
        """
        input:
            rgb_res, depth_res - tup, (width, height) e.g. (320, 240), (640, 480), (848, 480), (1280, 720)
            rgb_format, depth_format -- str, e.g. bgr8, z16 ... etc.
            fps -- int, frames per second

        e.g. 
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        doc of format you can take:
        https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.format.html#pyrealsense2.format.rgb8
        """
        self.RGB_RESOLUTION = rgb_res
        self.RGB_FORMAT = rgb_format
        self.DEPTH_RESOLUTION = depth_res
        self.DEPTH_FORMAT = depth_format
        self.FPS = fps
        self.config, self.profile, self.pipeline, self.depth_scale = None, None, None, None
        self._setup_config()
        self._setup_pipeline()
        self._get_depth_scale()
        self.get_config_info()
        # handler for aligning RGB and depth image
        self.align = rs.align(rs.stream.color)
        
    def _setup_config(self):
        config = rs.config()
        rgb_w, rgb_h = self.RGB_RESOLUTION
        depth_w, depth_h = self.DEPTH_RESOLUTION
        rgb_format = self._setup_format(self.RGB_FORMAT)
        depth_format = self._setup_format(self.DEPTH_FORMAT)
        fps = self.FPS
        config.enable_stream(rs.stream.depth, depth_w, depth_h, depth_format, fps)
        config.enable_stream(rs.stream.color, rgb_w, rgb_h, rgb_format, fps)
        self.config = config
        print('self.config is set!')

    def _setup_pipeline(self):
        assert self.config is not None, '[SETUP ERROR] self.config NOT PROPERLY SETUP'
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)
        self.pipeline = pipeline
        self.profile = profile
        print('self.pipeline and self.profile are set!')

    def _setup_format(self, format_str):
        assert format_str in ['bgr8', 'z16'], 'WRONG FORMAT INPUT: {}'.format(format_str)
        if format_str == 'bgr8':
            return rs.format.bgr8
        elif format_str == 'z16':
            return rs.format.z16
        else:
            print('IT SHOULDNT HAPPEN...')
            return None

    def _get_depth_scale(self):
        assert self.pipeline is not None, '[SETUP ERROR] self.pipeline NOT PROPERLY SETUP'
        assert self.profile is not None, '[SETUP ERROR] self.pipeline NOT PROPERLY SETUP'
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print('depth scale retrieved!')

    def get_config_info(self):
        print('\n########## RGB ##########')
        print('(w x h): {}'.format(self.RGB_RESOLUTION))
        print('format: {}'.format(self.RGB_FORMAT))
        print('\n########## DEPTH ##########')
        print('(w x h): {}'.format(self.DEPTH_RESOLUTION))
        print('format: {}'.format(self.DEPTH_FORMAT))
        print('scale: {}'.format(self.depth_scale))
        print('\n########## OTHERS ##########')
        print('fps: {}'.format(self.FPS))

    def test_streamline(self, frame_limit, is_process_depth = False, is_align = True):
        """
        streamline until # frame = frame_limit, apply depth postprocessing if is_process_depth = True
        """
        color_image, depth_image, depth_colormap = stream_camera(camera = self, frame_limit = frame_limit, 
                                                                 is_process_depth = is_process_depth, is_align = is_align)
        return color_image, depth_image, depth_colormap

    def get_snapshot_np(self, name, is_align = True):
        """
        take a snapshot from streamline (after warmup), and then output the snapshot (as numpy array)

        output:
            color_image -- np array, (height, width, channel) (uint 8)
            depth_image -- np array, (height, width, channel) (uint 8)
        """
        color_image, depth_image, depth_colormap = stream_camera(config = self.config, frame_limit = 1, is_process_depth= False, is_align = is_align)
        color_path = os.path.join('test', 'npy_test_case', name + '_rgb.npy')
        depth_1c_path = os.path.join('test', 'npy_test_case', name + '_d1c.npy')
        depth_3c_path = os.path.join('test', 'npy_test_case', name + '_d3c.npy')
        np.save(color_path, color_image)
        np.save(depth_1c_path, depth_image)
        np.save(depth_3c_path, depth_colormap)
        print('RGB SAVE: {}, {}'.format(color_image.shape, color_path))
        print('DEPTH 1C SAVE: {}, {}'.format(depth_image.shape, depth_1c_path))
        print('DEPTH 3C SAVE: {}, {}'.format(depth_colormap.shape, depth_3c_path))
        return color_image, depth_image

    def get_raw_frame(self, is_align = True):
        """
        act like an iterator for backend web app. determine RGBD frames alignment here

        output:
            rgb_frame -- pyrealsense2 frame instance
            depth_frame -- pyrealsense2 frame instance (1 channel)
        """
        frames = self.pipeline.wait_for_frames()
        if is_align:
            frames = self.align.process(frames)
        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        return rgb_frame, depth_frame


if __name__ == '__main__':
    name = 'cls20'
    resolution = (1280, 720)
    rs_handler = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
    rs_handler.test_streamline(frame_limit = 20)
    # color_image, depth_image, depth_colormap = rs_handler.test_streamline(
    #                                             frame_limit = 50, 
    #                                             is_process_depth = False)
    # color_path = os.path.join('test', 'npy_test_case', name + '_rgb.npy')
    # depth_1c_path = os.path.join('test', 'npy_test_case', name + '_d1c.npy')
    # depth_3c_path = os.path.join('test', 'npy_test_case', name + '_d3c.npy')
    # np.save(color_path, color_image)
    # np.save(depth_1c_path, depth_image)
    # np.save(depth_3c_path, depth_colormap)
    # print('RGB SAVE: {}, {}'.format(color_image.shape, color_path))
    # print('DEPTH 1C SAVE: {}, {}'.format(depth_image.shape, depth_1c_path))
    # print('DEPTH 3C SAVE: {}, {}'.format(depth_colormap.shape, depth_3c_path))
    # #color_image, depth_image = rs_handler.get_snapshot_np(name = 'cls1')
