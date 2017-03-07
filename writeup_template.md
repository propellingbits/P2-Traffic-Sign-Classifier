#**Traffic Sign Recognition** 

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

[image1]: ./writeup-images/visualization.jpg "Visualization"
[image2]: ./writeup-images/grayscale.jpg "Grayscaling"
[image3]: ./writeup-images/random_noise.jpg "Random Noise"
[image4]: ./writeup-images/ahead-only.jpg "Traffic Sign 1"
[image5]: ./writeup-images/construction.jpg "Traffic Sign 2"
[image6]: ./writeup-images/general-caution.jpg "Traffic Sign 3"
[image7]: ./writeup-images/keep-right.jpg "Traffic Sign 4"
[image8]: ./writeup-images/road-work.jpg "Traffic Sign 5"
[image9]: ./writeup-images/b4-pp.png "b4-pp"
[image10]: ./writeup-images/after-pp.png "after-pp"
[image11]: ./writeup-images/ddataset-v.png "ddataset-v"
[image12]: ./writeup-images/augData1.png "augData1"
[image13]: ./writeup-images/b4-Aug.png "b4-Aug"
[image14]: ./writeup-images/after-aug.png "after-aug"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/propellingbits/P2-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the seventh code cell of the IPython notebook.  

I used in-built functions\features of Python to calculate summary statistics of the traffic signs data set:

* The size of training set is 39209 images
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the eighth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart of training and test labels for each class ...

![alt text][image11]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

You can find my pre-processing code in cell #9. 

I have made it very configurable. It is very easy to turn on or off any preprocessing techniques in my program.

There is no dearth of options when it comes to preprocessing images. I will keep on exploring more options to get better results. 

I have included image normalization to clear out any light differences. Image sharperning filter helps in getting clear and brighter images. It is just pure fun to try different combinations of preprocessing techniques and see how the model behaves.

Here is an example of a traffic sign images before and after preprocessing with normalization enabled.

before images -
![alt text][image9]

after images -
![alt text][image10]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the ninth code cell of the IPython notebook.  

My final training set had 78117 images. My validation set and test set had 2526 and 12630 images.

The ninth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the original dataset is very unbalanced. There are classes which have thousands of images and there are also classes which have only couple hundred images. To add more data to the the data set, I used the following techniques - normalization and rotating images by certain angles. 

Here is an example of an original image and an augmented image:

Before augmentation -

![alt text][image13]

After augmentation -

![alt text][image14]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eleventh cell of the ipython notebook. 

I have used LeNet model with small changes to input and output layer. My final model consisted of the following layers:

| Layer         		    |     Description	        					            | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x3 RGB image   							            | 
| Convolution 5x5     	| 1x1x1x1 stride, valid padding,            	  |
|   					          | outputs 28x282x6								              |
| RELU                  | Activation                                    |
| Max pooling	      	  | 1x2x2x1 stride,  outputs 14x14x6 				      |
| Convolution 	        | 14x14x16 input, outputs 10x10x16				      |
| RELU                  | Activation                                    |
| Max pooling           | 1x2x2x1 stride, outputs 5x5x16                |
| Flatten               |                                               |
| Fully connected		    | 400 neurons								                    |
| RELU                  |                                               |
| Fully connected       | 120 input, outputs 84                         |
| RELU				          |           									                  |
| Fully connected       | 43 outputs maps to 43 traffic sign classes    |
|						            |												                        |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used 
1.  Adam optimizer for minimum tuning of learning rate and using less hyperparameters. I tried a lot of learning rates but found 0.001 to work great.
2. For each EPOCH, system prints the validation accuracy.
3. Training cycle stops after 10 epochs 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the thirtyfourth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 30% (29.7 %)
* training set with normalized and augmented data achieved accuracy of 98% (97.6 %)
* validation set accuracy of 87% (87.2%)
* test set accuracy of 87% (86.6%)

If a well known architecture was chosen:
* What architecture was chosen?
  LeNet Architecture
* Why did you believe it would be relevant to the traffic sign application?
  It is a well known and proven architecture. Its good use of convolution layers, max pooling, activation functions and fully connected layers made it a great choice for this project
* How does the final model's accuracy on the training, validation and test set provide evidence that the    model is working well?
  Both test and validation set have accuracy of 87%. It is a good number to believe that model is working well. Although, increasing the epochs can increase these numbers little bit more.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The speed limit images ('speed-limit.jpg', 'speed-limit2.jpg') are difficult to classify because there is some background scenary.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the thrityeighth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			            |     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only            | Ahead Only                                    |
| General Caution      	| General Caution								                | 
| Road Work     			  | Priority Road              										|
| Construction	        | Road Work                											|
| Stop    	      		  | Stop                        					 				|
|             			    |                                  							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 38th cell of the Ipython notebook.

For the first image, the model is very sure that this is a ahead-only sign (probability of 0.9), and the image is a ahead-only sign. The top five soft max probabilities were

| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .90         			    | ahead-only sign   									          | 
| .05     				      | Go straight or right			                    |
| .05					          | Go straight or left				                    |
|   	      			      |           					 				                  |
|   				            |                   							              |


For the second image, the model is 100% sure that it is a general caution sign which is true.


| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100         			    | general-caution sign   									      | 
     							              

For the third image, the model is 100% sure here also that it is a keep-right sign which is super accurate.


| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100         			    | keep-right sign   									          | 
     							              

For the fourth image, the model continued its success rate and predicted like it knows what it is doing. 

| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			    | priority road                                 |
| .10   								|	children crossing                             | 
     							              
For the fifth image, the model continued its winning streak and predicted like a boss. 

| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100         			    | road work                                     |
|        								|	                                              | 
  

For the sixth image, the model did a great job again and predicted like a champion. 

| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100         			    | stop                                          |
|        								|	                                              | 

For the seventh image, it was kind of expected that model will feel little turbulance here. Although, it was not expected that it will completely fail here, which is kind of disappointing. 

| Probability         	|     Prediction            	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100         			    | road-work sign   									            | 
     							              
