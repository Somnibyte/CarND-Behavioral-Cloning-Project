# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/backwards.gif "Backwards GIF"
[image2]: ./examples/center.gif "Track 1 GIF"
[image3]: ./examples/curves.gif "Curves GIF"
[image4]: ./examples/recover.gif "Recovery GIF"
[image5]: ./examples/track2.gif "Track 2 GIF"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on a model [published by the autonomous team at Nividia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The model consists of a normalization layer (model.py line 86-87), 5 convolutional layers (model.py lines 88-92), 
and 4 fully connected layers (model.py lines 93-109). 3 out of the 5 convolutional layers have a 5x5 filters and the rest have 3x3 filters. I slightly modified the network to include dropout and batch normalization layers alongside the fully connected layers of the network. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 97-109). 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114). I've also incorporated Batch Normalization layers to my model (model.py line 95-107). Recall that the first layer in my model consists of a normalization layer. The normalization layer in my model simply divides the image by 255.0. By normalizing the input features I'm effectively speeding up the learning process which is why I've added Batch Normalization layers to increase the speed of the learning process.

#### 4. Appropriate training data

My training dataset consisted of me driving 1 lap of first track while staying in the middle, driving only on the parts of the track that had curves, driving 1 lap on the new track while staying in the middle, and driving from the left and right lanes to the center of the road as a way to teach my model to recover. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I first started off with original Nividia model that was presented in the course. My initial validation loss readings were high and they would slightly increase during each epoch. This was a sign of overfitting. To combat this issue I experimented with adding dropout layers alongside each of my 4 fully connected layers. The dropout layers alleviated the overfitting issue, but they did not significantly reduce the loss so I focused on tuning the batch size and learning rate. I found that a batch size of 64 and a learning rate of 0.01 was sufficient in decreasing the loss by a substantial amount. I also experimented with applying batch normalization layers and I found that my loss continued to reduce slightly. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Image 							| 
| Cropping2D        | 61 pixels from the top, 25 pixels from the bottom
| Conv2D | 2x2 stride, filters = 24, kernel = 5 x 5, valid padding |
| RELU					|												|
| Conv2D | 2x2 stride, filters = 36, kernel = 5 x 5, valid padding |
| RELU					|												|
| Conv2D | 2x2 stride, filters = 48, kernel = 5 x 5, valid padding |
| RELU					|												|
| Conv2D | 2x2 stride, filters = 64, kernel = 3 x 3, valid padding |
| RELU					|												|
| Conv2D | 2x2 stride, filters = 64, kernel = 3 x 3, valid padding |
| RELU					|												|
| Fully connected		| Output = 1000        									|
| Batch Normalization					|												|
| RELU					|												|
| Dropout					|					50% Chance			|
| Fully connected		| Output = 100        									|
| Batch Normalization					|												|
| RELU					|												|
| Dropout					|					50% Chance			|
| Fully connected		| Output = 50       									|
| Batch Normalization					|												|
| RELU					|												|
| Dropout					|					50% Chance			|
| Fully connected		| Output = 10        									|
| Batch Normalization					|												|
| RELU					|												|
| Dropout					|					50% Chance			|
| Fully connected		| Output = 1        									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 1 lap on track one using center lane driving. Here is an example GIF of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid the riding on top of the lane lines. This GIF represents the vehicle recovering from the right lane. :

![alt text][image4]

Then I recorded the car driving on the curved parts of track one:

![alt text][image3]

This was followed by recording the car driving on track one going backwards.

![alt text][image1]

Finally, I recorded myself driving the car on track two for 1 lap.

![alt text][image5]

I augmented my dataset by flipping the center images. After the collection process I had over 15,000 data points. 


