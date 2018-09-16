import argparse
import base64
from datetime import datetime
import os
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
import image_processing as ip

sio = socketio.Server()
app = Flask(__name__)
model = None
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)
            # image = ip.preprocessing_NVIDIA(ip.Image(image))
            image = preprocess(ip.Image(image))
            image = np.array([image.img])

            steer = float(model.predict(image, batch_size=1))

            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steer ** 2 - (speed / speed_limit) ** 2

            print('{} {} {}'.format(steer, throttle, speed))
            send_control(steer, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


def main():
    parse = argparse.ArgumentParser(description='Autonomous driving')
    parse.add_argument('model', type=str, help='Path to .h5 model file.')
    parse.add_argument('-1', help='choose model_1 preprocessing', dest='model_1', type=bool, default=False)

    args = parse.parse_args()

    global preprocess

    if args.model_1:
        print('Model_1 preprocessing')
        preprocess = ip.preproc_model_1
    else:
        print('NVIDIA model preprocessing')
        preprocess = ip.preprocessing_NVIDIA


    global model
    model = load_model(args.model)
    global app
    global sio
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == '__main__':
    main()
