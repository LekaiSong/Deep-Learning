Deep-Learning
=============
#### The image data used in this miniproject can be downloaded from the following link: http://download.tensorflow.org/example_images/flower_photos.tgz

We only use two kinds of flower images for our training. Thus, take roses and sunflowers as an example and meanwhile delete other 3 kinds of flower images. The whole files structure is as follows.

>./Deep-Learning/
>>#### ... <br>training_set
>>>roses <br> sunflowers
>>#### test_set
>>>roses <br> sunflowers
>>#### training_dataset
>>>roses <br> sunflowers
>>#### test_dataset
>>>roses <br> sunflowers
>>#### Ps: training_dataset and test_dataset are the folders where resized images are stored. 

#### The first two steps below are trying explaining how to train a model by yourself, but if you only want to demo my pre-trained model the first two steps could be skipped.
>1) run resize299.py in the terminal. For the reason that inceptionV3 model requires standard input format (size of images), we need to resize those images from both training_set and test_set into 299pixel*299pixel by default. The same as resize64.py.

>2) Secondly, run inception.py in the terminal to train the model. It's a time-consuming process which depends on what CPU and GPU you are using. Values in line from #29 to #31 represent the number of classes, number of epochs and batch size, which can also be modified as you like. As soon as it's completed, the graphs of the connection between epochs and training accuracy would be shown on the screen clearly. The trained model is saved as 'inceptionV3.model' at the same time.

#### 3. 'python inceptionpredict.py -i path/test_image'. It can predict what class the image you test belongs to. The output are the test image and the possibility of prediction shown as a graph. 
<br> Take an example. 
>![inception_test_result](https://github.com/LekaiSong/Deep-Learning/blob/master/inception_prediction_result.png)

#### 4. The CNN.py provides a much more convenient training model because only a few typical layers such as convolutional layer and pooling layer are added. In this model, RGB images are resized into 64pixel*64pixel.

### Comparison

When the scale of training_dataset is not that big, you can use simple CNN model instead of InceptionV3 one to get a fast and high-accuracy prediction result (both models' test accuracies are over 90%). However, when it comes to a large scale dataset, we prefer mature training models (though it takes more time) since their layers structure has been already optimized so that they are capable of dealing with complex situations and filtering more details.
