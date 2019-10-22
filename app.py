"""
main function to launch web app server

STEPS
1a. one window output
1b. speed up the loop
2. separate window output
3. add widget 
4. add CSS stylesheet

REFERENCE
1. why cv2.imencode takes so long? https://answers.opencv.org/question/207286/why-imencode-taking-so-long/

LOG
[20/10/2019]
- setting app.run_server(debug = True) will make RGBDhandler() setup twice, caused by reloading functionality
- jpeg encoding causes a speed bottleneck (i.e. cv2.imencode())

[21/10/2019]
- How to generate a split video stream?
- Stream faster when in charge
"""
import os
import sys
import time

import numpy as np
import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask, Response
from turbojpeg import TurboJPEG

import d435_module
from camera_config import RGBDhandler
from img_stream import process_frame

# setup realsense camera and server
resolution = (1280, 720)
camera = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
# jpeg encoding optimiser
jpeg = TurboJPEG()

server = Flask(__name__)
app = dash.Dash(__name__, server=server)


def single_gen(camera):
    while True:
        print(time.time())
        rgb_frame, depth_frame = camera.get_raw_frame()
        if rgb_frame is None or depth_frame is None:
            print('blank frame received, retrieve the next frame')
            continue
        rgb_img, depth_img, depth_colormap = process_frame(rgb_frame, depth_frame)
        display_img = np.concatenate((rgb_img, depth_colormap), axis = 1)
        # optimise jpeg encoding
        jpeg_encode = jpeg.encode(display_img)
        #jpeg_rgb = jpeg.encode(rgb_img)
        #jpeg_d = jpeg.encode(depth_colormap)
        #_, jpeg = cv2.imencode('.bmp', display_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_encode + b'\r\n\r\n')
        #yield b'--frame\r\n'


# bind a function to a URL
@server.route('/video_feed')
def video_feed():
    a = Response(single_gen(camera), mimetype = 'multipart/x-mixed-replace; boundary=frame')
    return a


app.layout = html.Div(children = [
    html.Nav(className = 'nav', children = [
        html.A('App1', className = 'nav-app1', href = '/nav-app1'),
        html.A('App2', className = 'nav-app2', href = '/nav-app2')
    ]),
    html.H1("Webcam Test"),
    html.Img(src = "/video_feed")
])


if __name__ == '__main__':
    app.run_server(debug = False)
