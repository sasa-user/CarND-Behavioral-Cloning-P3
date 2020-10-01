# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./writeup_img/center_2016_12_01_13_31_14_702.jpg "img1"
[image3]: ./writeup_img/left_2016_12_01_13_31_14_702.jpg "img2"
[image4]: ./writeup_img/right_2016_12_01_13_31_14_702.jpg "img3"

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

My model consists of a convolution neural network layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 67-75) 

The model includes RELU layers to introduce nonlinearity after almost every conv layer (for example code line 68), and the data is normalized in the model using a Keras lambda layer (code line 64).

#### 2. Attempts to reduce overfitting in the mode

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 91). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning 

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data 

Training data that I used was provided with the workspace. I did run the simulator to try it out myself. For creating data I would go with recomendations from the lessons, so combination of center lane driving, recovering from the left and right sides, driving in the oposit direction...

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with design that already exists. 

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because it is simple enough to implement and it is proven in real driving tests.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had trouble 
keeping in track.
To combat that I had to use generators to train with much more data and process data better and I provided images with different camera angle, cropping and also added fliping.

There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I had to fix epochs, add max pooling and few of acitvation layers, since I wanted tricky parts to be better processed.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a table visualization of the architecture:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================

#### 3. Creation of the Training Set & Training Process

As I mentioned is some section before, I used provided data for training my network.

While I was preprocessing data, I had in mind data collecion tips that we received in our lessons. So, with that, I tried to process data so that I have multiple angles in my set for recovery (left and right - with adjusted measurements), to have images with nice center, to flip images so I could have counter clockwise driving (though I didn't use this at the end since there was no much improvment in driving), and I adjusted brightness to some images so I could maybe better differentiate round from other parts of the track. 

Here is an example image of center lane driving, and left and right angle views:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I finally used my data generator method for training and validation sets. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs that I ended up using is 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Video of the simulation run: [video.mp4](https://drive.google.com/file/d/1xZnV7Dl4s7u-28KNpDTS8ZJu-GA51WiH/view?usp=sharing)