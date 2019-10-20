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

import d435_module
from camera_config import RGBDhandler
from img_stream import process_frame

# setup realsense camera and server
resolution = (1280, 720)
camera = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
server = Flask(__name__)
app = dash.Dash(__name__, server=server)


def gen(camera):
    while True:
        print(time.time())
        rgb_frame, depth_frame = camera.get_raw_frame()
        if rgb_frame is None or depth_frame is None:
            print('blank frame received, retrieve the next frame')
            continue
        rgb_img, depth_img, depth_colormap = process_frame(rgb_frame, depth_frame)
        display_img = np.concatenate((rgb_img, depth_colormap), axis = 1)
        #_, jpeg = cv2.imencode('.bmp', display_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + display_img.tobytes() + b'\r\n\r\n')
        #yield b'--frame\r\n'

# bind a function to a URL
@server.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype = 'multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div(children = [
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])


if __name__ == '__main__':
    app.run_server(debug = False)
