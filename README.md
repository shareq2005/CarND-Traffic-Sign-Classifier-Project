# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goal of this project is to apply deep learning techniques using TensorFlow to build a german traffic sign classifier. A convolutional neural network was built, and then trained with a data-set consisting of a large number of traffic sign images. After that, some new images were downloaded from the net to see how well the classifier predicts the new images.

[//]: # (Image References)

[image1]: ./images/dataset_visualization.png "Visualization"
[image2]: ./images/model_architecture.png "Model Architecture"
[image3]: ./images/example_images.png "Example Images"
[image4]: ./images/prediction_table.png "Prediction Table"
[image5]: ./images/softmax_incorrect_prediction.png "Softmax for Incorrect Prediction"

## Dataset Summary and Exploration

I used Python and numpy to provide the summary statistics of the traffic signs data set:
* Number of training examples = 34799
* Number of testing examples  = 12630
* Shape of a traffic sign image = (32, 32, 3)
* Number of classes = 43

The following is a visualization of the data set:

![alt text][image1]

These are the bar charts of the training, validation & testing sets. The plot is of the traffic sign index against the percentage of their occurrence in the data set. As you can see from the plots above, the pattern across the three data sets is fairly similar, i.e. some traffic signs have a higher number of occurrence as compared to others. For example, the traffic signs corresponding to indexes 1, 2 & 12 have a higher occurrence ( above 5 % total occurrence in each of the three data-sets). These indexes correspond to the signs below:
* Index 1   = Speed Limit (30km/h)
* Index 2   = Speed Limit (50km/h)
* Index 12 = Priority road

---
## Design and Test a Model Architecture

### Preprocessing the dataset
Each of the traffic signs were preprocessed by converting to grayscale, and then normalizing the pixel values. 
 
I used a grayscale method since the traffic signs are mostly dependent on their shape as opposed to their color. Therefore, converting to grayscale allows the network to focus more on the shape of the traffic sign. The grayscale conversion was done through the OpenCV cvtColor method.
 
I then normalized the image using a basic normalization calculation - For each pixel, the value was adjusted using the formula ( pixel value = ( pixel value - 128 )/128 ). Normalization is done so that all the feature values have a similar range.

### Final Model Arhictecture

![alt text][image2]

The model above was built using extending the LeNet architecture. LeNet’s architecture is a good proven design which is used to classify numbers. Basically, it is trained to identify patterns in images. The problem in this case is to classify the traffic signs, which for the most part are simply different patterns on images. Therefore, the LeNet architecture was chosen as a base design for this task.

### Model Training
The optimizer chosen was the AdamOptimizer which is gradient-descent optimizer. The batch size used was 128, and the number of epochs set to 50. The learning rate was set to 0.001.

### Solution Approach
Without any modification, I found that the average accuracy using LeNet architecture was around 88%. Therefore, the LeNet architecture did need some modifications. Clearly, the architecture was underfitting the traffic sign classification. LeNet was designed for a smaller number of classes. Therefore, the network needed to be larger to allow for more number of classes, i.e. 43. The number of filters were increased in the two convolutional layers, and all the fully connected layers were widened. This allowed the network’s accuracy on the validation set to improve to 95.9%. To prevent overfitting, a dropout layer was added after each of the convolutional and max-pooling layers. The final accuracy on the training, validation & testing set are as follows:
 
* Training Set Accuracy - 99.9 %
* Validation Set Accuracy - 95.9%
* Testing Set Accuracy - 95.7 %

### Test Model on New Images
To test how the classifier performs on real world data, the  following images were downloaded from the internet. The images were resized to 32x32 pixels (which is the input size of our network). 

![alt text][image2]

Reasons why each of the images above may be difficult to classify:
* Image 0.png - (Max Speed 50 km/h) - Not difficult to classify since 50km/h is the most common in the most common image in our data-set (over 5% of images in both the training & validation sets).
* Image 1.png - (Priority road) - Difficult due to the straight edges in the background
* Image 2.png - (Max Speed 60 km/h) - Not difficult since there are enough images in the data-set for this sign.
* Image 3.png - (Keep left) - Difficult to classify due to the added noise. It was taken from Getty Images, and therefore includes their watermark.
* Image 4.png - (Children Crossing) - Difficult to classify since the image is taken from a bad angle and appears a bit distorted.
* Image 5.png - (Max speed 30 km/h) - Not difficult to classify since this sign comprises over 5% of the data-set.
* Image 6.png - (Children Crossing) - Not difficult to the added trees in the background.
* Image 7.png - (Ahead only) - Not difficult to classify due to the image being dim.
* Image 8.png - (Bumpy road) - Not difficult to classify due to trees in the background
* Image 9.png - (Yield) - Difficult due to the circular shapes around the image
 
The same pre-processing steps were applied ( gray-scaling and normalizing ). Using our trained model, the accuracy on these images was 90 %. Here are the results of the prediction:

![alt text][image3]

The test images include two images of the ‘Children crossing’ sign. One of them was labeled incorrectly. The image which is incorrectly detected was taken from a bad angle, and appears a bit distorted. Perhaps augmenting the data with some distortions would have allowed our model to take this into account. If we look at the softmax probabilities for this prediction as follows, we can see that the model wasn’t too sure about the certainty of the incorrect traffic index it predicted. The incorrect index 1 (which is for ‘Speed Limit 30km/h’) only had a probability of 0.1355 ( around 13 %).

![alt text][image4]

