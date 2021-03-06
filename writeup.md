# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class_counts]: ./class_counts.png "Class Counts"
[pre_pre_processing]: ./graphs/valid_pre_processing.png "Before"
[post_pre_processing]: ./graphs/valid_post_processing.png "After"

[graph_img_1]: ./graphs/img_1.png "Right Of Way Sign"
[graph_img_2]: ./graphs/img_2.png "Keep Right"
[graph_img_3]: ./graphs/img_3.png "30kmph"
[graph_img_4]: ./graphs/img_4.png "60kmph"

[cropped_img_1]: ./own_examples/cropped/100_1607.jpg "a"
[cropped_img_2]: ./own_examples/cropped/459380917.jpg "a"
[cropped_img_3]: ./own_examples/cropped/459381023.jpg "a"
[cropped_img_4]: ./own_examples/cropped/459381091.jpg "a"
[cropped_img_5]: ./own_examples/cropped/459381275.jpg "a"
[cropped_img_6]: ./own_examples/cropped/mifuUb0.jpg "a"
[cropped_img_7]: ./own_examples/cropped/469763319.jpg "a"
[cropped_img_8]: ./own_examples/cropped/465921901.jpg "a"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate the following summary statistics of the traffic signs data set:

- 34'799 training examples
- 12'630 testing examples.
- The images are 32 pixels wide and high, and contain three channels (RGB)
- There are 43 unique classes in the set.

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the "Include an exploratory visualization of the dataset" of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of each class contained in the training set.

![alt text][class_counts]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the "Pre-process the Data Set" section of the IPython notebook.

As a first step, I normalized the images. Normalization is documented to reduce convergance time whilst training the neural net. This step was done using the following code.

    2*image.astype(np.float32)/255.0 -1

The images were then sharpened through Scipys built in sharpen filter. Sharpening the filters meant that the edges in many of the blurry images were more clearly defined, and hence easier to detect. It was done using the following code.

    scipy.misc.imfilter(image, 'sharpen')

Here is an example of the preprocessing

##### Before Pre-processing

![alt text][pre_pre_processing]

##### After Pre-processing

![][post_pre_processing]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Fortuantly splitting the code into training, validation and test data was not necessary, as the dataset provided by Udacity had already been split up!

Augmenting the dataset, focusing on rotation and translation, was explored, but after ultimately dropped, after it was found to reduce accuracy. Instead I decided to focus on improving the model design.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the "Model Architecture" section of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU |  |
| Convolution 1x1 | 1x1 stride, outputs 28x28x32 |
| Maxpool 2x2 | 2x2 stride, outputs 14x14x32 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x32 |
| RELU |  |
| Convolution 1x1 | 1x1 stride, outputs 10x10x32 |
| Maxpool 2x2 | 2x2 stride, outputs 5x5x32 |
| Flatten | Outputs 800 |
| Fully Connected | Outputs 120 (25% dropout) |
| RELU | |
| Fully Connected | Outputs 84 (25% dropout) |
| RELU | |
| Fully Connected | Outputs 43 |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the "Train, Validate and Test the Model" section of the iPython notebook.

To train the model, I used an Adam Optimizer with a base learning rate of 0.001, over 25 epochs and a batch size of 128. Mu and sigma were set to 0 and 0.1 respectively, while dropout was set to 25% for all dropout layers.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the "Analyze Performance" section of the Ipython notebook.

My final model results were:
* training set loss of 0.003
* validation set accuracy of exactly 93%
* test set accuracy of 75%

I first attempted to reuse the LeNet architecture that was used in the previous labs, but did not get a sufficiently high validation set accuracy.

After some research, I attempted to reimplement Google's inception model. Unfortuantly I was unable to successfully achieve model convergance in a timely manner (45% validation accuracy at 5 minutes on a GTX 970).  

Eventually I decided to go back to the LeNet model and attempt to modify it to acheive greater results.

These modifications included adding dropout (in order to aid generalization) and maxpooling layers (in order to reduce the parameter count and increase training speed).  

Using the LeNet model was a good choice. It's simplicity means I could train and run it quickly, allowing me to iterate on the design quickly. It was a good fit to the traffic sign application as it was originally designed for MNIST, a similiar image recognition challenge on a relatively small number of classes. The fact that it is able to achieve a 93% validation accuracy is a sign that it works well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![a][cropped_img_1]
![alt text][cropped_img_2]
![alt text][cropped_img_3]
![alt text][cropped_img_4]
![alt text][cropped_img_5]
![alt text][cropped_img_6]
![alt text][cropped_img_7]
![a][cropped_img_8]

The first image is a relatively simple one.

The second image is missing the bottom of the sign. This might stress the network if it cant find a perfect circle.

The third image has a water mark.

The fourth image has a very slight upward tilt.

The fifth image is also a relatively easy one.

The sixth image has a large amount of glare on the sign

The seventh image has a watermark on the image.

The eighth image has a small amount of glare, and a watermark.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Right-of-way at the next intersection|Right-of-way at the next intersection |
|Keep right|Keep right |
|Speed limit (30km/h)|Speed limit (30km/h) |
|Speed limit (30km/h)|Speed limit (60km/h) |
|Right-of-way at the next intersection|Right-of-way at the next intersection|
|Children crossing|Children crossing|
|Traffic signals|Pedestrians|
|Road work |Road work|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This compares unfavorably to the accuracy on the test set of 99.9%. It's possible that we have issues because of the small sample size, or that we overfitted to the test data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Output Top 5 Softmax Probabilities For Each Image Found on the Web" of the Ipython notebook.

For the first image, the neural net is relatively sure that this is a right of way sign (close to double its next prediction of a snow warning). The next predictions are can be seen below.

![alt text][graph_img_1]

For the second image, it is also quite sure it has the correct answer.

![alt text][graph_img_2]

All of the predictions made for this speed sign are the different classes of speed sign. It correctly picks out the first one, though it is relatively unsure. Its possible it only predicted the correct sign because its one of the more common sign types.

![alt text][graph_img_3]

This holds true for this sign. It did not guess the 60kmph sign at all, but did guess it was either 50kmph or 30kmph. Those classes are the most common in the training set, so it is natural that it would pick it. Greater accuracy could be achieved by creating a seperate neural network to guess speed signs exclusively, or by evening out the ratio of speed limit signs.

![alt text][graph_img_4]
