#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

#parse arguments
parser=argparse.ArgumentParser()
parser.add_argument('-i','--image',required=True,help="path to resized test image")
args = parser.parse_args()
    
# size of target image
target_size = (299, 299) #fixed size for InceptionV3 architecture
 
# prediction
def predict(model, img, target_size):
  """
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
 
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]
 
# draw 
labels = ("roses","sunflowers")
def plot_preds(image, preds,labels):
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  plt.barh([0,1], preds, alpha=0.5)
  plt.yticks([0,1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()
 
# load model
model = load_model('InceptionV3.model')
 
# test
img = Image.open(args.image) #test image you like
out = img.resize((299,299),Image.ANTIALIAS) #resize image with high-quality
out.save(r'./inceptiontest.jpg', 'jpeg')
img = Image.open('inceptiontest.jpg')
preds = predict(model, img, target_size)
print(preds)
plot_preds(img, preds,labels)
print("prediction finished")