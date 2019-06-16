import cv2 as cv
import numpy
from keras.applications.nasnet import NASNetMobile, decode_predictions, preprocess_input
from keras.preprocessing import image

vid = cv.VideoCapture(0)
model = NASNetMobile(input_shape=None, include_top=True, weights=None)

model.load_weights('h5/NASNet-mobile.h5')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


def resize_img(img, width=None, height=None, inter=cv.INTER_AREA):
    h, w = img.shape[:2]

    if width is None and height is None:
        return img
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    resized = cv.resize(img, dim, interpolation=inter)
    return resized


i = 1
while True:
    _, frame = vid.read()

    img = resize_img(frame, width=224, height=224)

    # h, w = img.shape[:2]
    x = image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)

    print("lan: " + i.__str__())
    print(decode_predictions(result, top=10))
    print("------------------------------------------------------------------------")

    cv.imshow('Classification Real-time Video', img)

    key = cv.waitKey(1) & 0xFF
