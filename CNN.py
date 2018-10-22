#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()  
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation = 'relu')) 
#(64,64) means 64*64 pixels,3 means RGB, relu means activate
classifier.add(MaxPooling2D(pool_size = (2,2))) 
#first pooling: reduce filesize to reduce checkpoint of next layer
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#second pooling
classifier.add(Flatten())
#transferred into one-dimension vector 
classifier.add(Dense(units = 128, activation='relu')) 
#connected layer
classifier.add(Dense(units=1, activation='sigmoid'))
#initialize the output layer, for 2-sort classification, units=1
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
#compile classifier

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range = 0.2, horizontal_flip = True) 
#image augmentations.
test_datagen = ImageDataGenerator(rescale = 1./255) 
training_set= train_datagen.flow_from_directory('./training_dataset', target_size=(64, 64), batch_size=10, class_mode='binary') 
test_set=test_datagen.flow_from_directory('./test_dataset', target_size=(64, 64), batch_size=10, class_mode='binary')
#batch_size means pick up how many images for per training
classifier.fit_generator(training_set, steps_per_epoch =500, epochs = 5, validation_data =test_set, validation_steps = 2000) 
#one epoch means to train all images once, steps_per_epoch means the amount of images in training set.
  
import numpy as np 
from keras.preprocessing import image
from PIL import Image
img = Image.open('./u=2704567099,2328751227&fm=26&gp=0.jpg') #test image you like
out = img.resize((64,64),Image.ANTIALIAS) #resize image with high-quality
out.save(r'./test.jpg', 'jpeg')
test_image = image.load_img('test.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis=0) 
result = classifier.predict(test_image) 
training_set.class_indices 
if result[0][0] == 1: 
    prediction = 'sunflowers' 
else: 
    prediction ='roses' 
print(prediction)
