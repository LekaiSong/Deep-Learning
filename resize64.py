#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os

def is_img(ext):
    ext = ext.lower()
    if ext == '.jpg':
        return True
    elif ext == '.jpeg':
        return True
    else:
        return False

training_set_path = "./training_set/roses/"
training_images1 = os.listdir(training_set_path)
#print(training_images1)
i=0
for training_image1 in training_images1:
    #print(training_image1)
    if is_img(os.path.splitext(training_image1)[1]): #judge if file is image types
        img = Image.open(os.path.join(training_set_path,training_image1)) 
        #print(img)
        out = img.resize((64,64),Image.ANTIALIAS) #resize image with high-quality
        #print(out)
        out.save(r'./training_dataset/roses/%03d.jpg'%i, 'jpeg')
        i += 1
        
training_set_path = "./training_set/sunflowers/"
training_images2 = os.listdir(training_set_path)
#print(training_images2)
j=0
for training_image2 in training_images2:
    #print(training_image2)
    if is_img(os.path.splitext(training_image2)[1]):
        img = Image.open(os.path.join(training_set_path,training_image2))
        #print(img)
        out = img.resize((64,64),Image.ANTIALIAS) #resize image with high-quality
        #print(out)
        out.save(r'./training_dataset/sunflowers/%03d.jpg'%j, 'jpeg')
        j += 1
        
test_set_path = "./test_set/roses/"
test_images1 = os.listdir(test_set_path)
#print(test_images1)
a=0
for test_image1 in test_images1:
    #print(test_image1)
    if is_img(os.path.splitext(test_image1)[1]):
        img = Image.open(os.path.join(test_set_path,test_image1))
        #print(img)
        out = img.resize((64,64),Image.ANTIALIAS) #resize image with high-quality
        #print(out)
        out.save(r'./test_dataset/roses/%03d.jpg'%a, 'jpeg')
        a += 1

test_set_path = "./test_set/sunflowers/"
test_images2 = os.listdir(test_set_path)
#print(test_images2)
b=0
for test_image2 in test_images2:
    #print(test_image2)
    if is_img(os.path.splitext(test_image2)[1]):
        img = Image.open(os.path.join(test_set_path,test_image2))
        #print(img)
        out = img.resize((64,64),Image.ANTIALIAS) #resize image with high-quality
        #print(out)
        out.save(r'./test_dataset/sunflowers/%03d.jpg'%b, 'jpeg')
        b += 1

print("resize finished")

