# -*- coding: utf-8 -*-

from datetime import datetime

print("1. Importing components, time:", datetime.now())
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from keras.preprocessing import image


from keras.models import load_model

model = load_model('model.h5')
model.summary()


test_image = image.load_img("TestDog.png", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = mtcn.predict(test_image)
print ("########## Prediction ########")
print (result[0][0])
