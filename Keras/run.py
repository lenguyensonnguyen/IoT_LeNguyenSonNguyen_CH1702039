from keras.applications.nasnet import NASNetMobile, decode_predictions, preprocess_input
from keras.preprocessing import image
import numpy
import os

model = NASNetMobile(input_shape=None, include_top=True, weights=None)

model.load_weights('h5/NASNet-mobile.h5')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

data_path = 'data'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

for img_name in img_list:
    if img_name == '.gitkeep':
        continue
    img = image.load_img(os.path.join(data_path, img_name), target_size=(224, 224))
    x = image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)

    print(img_name + ": ")
    print(decode_predictions(result, top=10))
    print("------------------------------------------------------------------------")
