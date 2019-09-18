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

#from keras.models import load_model
#model = load_model('model.h5')
#model.summary()


print("2. Initialising the CNN, time:", datetime.now())
classifier = Sequential()

print("3. Convolution, time:", datetime.now())
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

print("4. Pooling, time:", datetime.now())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

print("5. Adding a second convolutional layer, time:", datetime.now())
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

print("6. Adding pooling for second convolutional layer, time:", datetime.now())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

print("7. Flattening, time:", datetime.now())
classifier.add(Flatten())

print("8. Full connection, time:", datetime.now())
classifier.add(Dense(output_dim = 128, activation = 'relu'))

print("9. Adding sigmoid activation function for final layer of NN, time:", datetime.now())
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

print("10. Compiling the CNN, time:", datetime.now())
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


print("11. Fitting the CNN to the images, time:", datetime.now())
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

print("12. Training starts, time:", datetime.now())
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

print("13. Training complete, time:", datetime.now())

#classifier.save("model.h5")