import json

import cv2 as cv

from Retinex import retinex

vid = cv.VideoCapture(0)


def resize_img(img, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    h, w = img.shape[:2]

    if width is None and height is None:
        return img
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(img, dim, interpolation=inter)
    return resized


with open('config.json', 'r') as f:
    config = json.load(f)

while True:
    _, frame = vid.read()

    img = resize_img(frame, width=200)

    img = retinex.MSRCP(img, config['sigma_list'],
                        config['low_clip'],
                        config['high_clip'])

    # h, w = img.shape[:2]

    cv.imshow('Enhanced Real-time Video', img)

    key = cv.waitKey(1) & 0xFF
