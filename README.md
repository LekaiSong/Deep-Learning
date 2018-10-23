# Deep-Learning
The image data used in this miniproject can be downloaded from the following link: http://download.tensorflow.org/example_images/flower_photos.tgz

We only use two kinds of flower images for our training. Thus, take roses and sunflowers as an example and meanwhile delete other 3 kinds of flower images. The whole files structure is as follows.

./data/

(files)

	--CNN.py

	--resize.py

	--inception.py

	--inceptionprediction.py

(folders)

	--training_set
	
		--roses
	
		--sunflowers

	--test_set
	
		--roses
	
		--sunflowers

Firstly, run resize.py in the terminal. For the reason that inceptionV3 model requires standard input format (size of images), we need to resize those images from both training_set and test_set into 299pixel*299pixel, or whatever size you prefer (if so values in row #25, #39, #53 and #67 (4 places in total) should be modified.

Secondly, run inception.py in the terminal to train the model. It's a time-consuming process which depends on what CPU and GPU you are using. Values in row from #29 to #31 represent the number of classes, number of epochs and batch size, which can also be modified as you like. As soon as it's completed, the graphs of the connection between epochs and training accuracy would be shown on the screen clearly. The trained model is saved as 'inception.model' at the same time.

Then, run inceptionprediction.py to test how it works for test_set. The model can predict what class the image you want to test belongs to. What you need to do is to change the iamge path in row #49. The output is the possibility of prediction which is shown as a graph as well.

Additionally, the CNN.py provides a much more convenient training model because only a few typical layers such as convolutional layer and pooling layer are added. In this model, RGB images are resized into 64pixel*64pixel.

Comparison：

When the scale of training_set and test_set is not that big, you can use this simple model instead of former mature one to get a fast and high-accuracy prediction result. However, when it comes to a large scale dataset, we should pay more attention to mature training models (though it takes lots of time) since their layers structure has been already optimized so that they are capable of dealing with complex situations and filtering more details.
