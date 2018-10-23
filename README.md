# Deep-Learning
The image data used in this miniproject can be downloaded from the following link: http://download.tensorflow.org/example_images/flower_photos.tgz

We only use two kinds of flower images for our training. Thus, Take roses and sunflowers as an example and meanwhile delete other 3 kinds of flower images. The whole files structure is as follows.

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

Firstly, run resize.py in the terminal. For the reason that tensorflow requires standard input format (size of images), we need to resize those images from both training_set and test_set into 64pixel*64pixel or whatever size you prefer.

Secondly, 
