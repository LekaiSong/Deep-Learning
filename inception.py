#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
 
def get_nb_files(directory):
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt
 
# data preparation
IM_WIDTH, IM_HEIGHT = 299, 299 #image size that InceptionV3 requires
FC_SIZE = 1024                # number of fully-connected checkpoint
NB_IV3_LAYERS_TO_FREEZE = 172  # number of freeze layer
  
train_dir = './training_dataset'  # training set
val_dir = './test_dataset' # validation set
output_model_file = './InceptionV3.model'
nb_classes= 2
nb_epoch = 10
batch_size = 50
 
nb_train_samples = get_nb_files(train_dir)      # number of training set
nb_classes = len(glob.glob(train_dir + "/*"))  # number of classification
nb_val_samples = get_nb_files(val_dir)       #number of validation set
nb_epoch = int(nb_epoch)                # number of epoch
batch_size = int(batch_size)           
 
# image generator
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
 
# training set and validation set
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,class_mode='categorical')
 
validation_generator = test_datagen.flow_from_directory(
val_dir,
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,class_mode='categorical')
 
# add new layer
def add_new_last_layer(base_model, nb_classes):
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model
# freeze the layers before NB_IV3_LAYERS
def setup_to_finetune(model):
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
 
# networking structure
model = InceptionV3(weights='imagenet', include_top=False)
model = add_new_last_layer(model, nb_classes)
setup_to_finetune(model)
 
# training model
history_ft = model.fit_generator(
train_generator,
samples_per_epoch=nb_train_samples,
nb_epoch=nb_epoch,
validation_data=validation_generator,
nb_val_samples=nb_val_samples,
class_weight='auto1')
 
# save model
model.save(output_model_file)
 
# draw
def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()
 
# draw acc-loss relation 
plot_training(history_ft)
