**Behavioral Cloning Project**

The following files were submitted:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results


[//]: # (Image References)

[data_distr]: ./figure_1.png "Data distribution after augmentation"
[blurred_image]: ./figure_2.png "Blurred image"
[blurred_image_flipped]: ./figure_3.png "Flipped Blurred image"
[cropped_image]: ./figure_4.png "Cropped image in BGR color space"

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model has the following architecture (model.py lines 153-185) including 2 convolution layers, 2 maxpool layers, 3 elu activation layers, and two fully connected layers:

- Input (160x320x3)
- Cropping 40 from above, 15 from below (above horizon, and front of car)
- Normalization Lambda layer
- Conv Layer (2x2x2 filter, stride 1)
- Elu activation
- Maxpool (2x2 kernel, 2x2 stride)
- Conv Layer (4x2x2 filter, stride 1)
- Maxpool (2x2 kernel, 2x2 stride)
- Elu activation
- Flatten
- Fully connected layer to 32 nodes
- Elu activation
- Fully connected layer (32 -> 1)


#### 2. Attempts to reduce overfitting in the model

To reduce overfitting the model was constructed with not too many layers, l1 regularization (lines 176, 179) was introduced, and images were blurred and darkened during training (lines 31, 32).  Also the data set was split into training and validation sets and the epoch with small validation mse was chosen to avoid overfitting (the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track).

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate of 0.0001 (model.py line 188).

#### 4. Appropriate training data

The data was loaded from three datasets using generators (line 96). The Udacity dataset was the most stable since my simulator was running at lowest settings due to computer limitations.

I generated two additional datasets with the simulator.  One, where the car is purposefully driven straight until it gets close to the lane edge and then is recovered and so on, this allows to avoid constantly pushing record button.  The problem is that the straight driving data is bad driving, so I only read nonzero angles, this also balances the data.  The resulting data is left biased, but is flipped as part of the training.
The second simulation generated dataset has additional data now with good driving so the zero angles are included.

### Strategy

My first idea was to use a simple linear Flatten layer with only 10 images and 20 epochs to make sure that the pipeline works and the model produces predictions.

This resulted in a small frustration as I was trying to understand why my training loss was higher than validation loss when using 20 epochs, since it seemed a bit counter-intuitive as I was expecting complete overfitting with only 10 images.  After going through the Keras documentation I convinced myself that the discrepancy was due to 1) the fact that training and validation losses are calculated differently when multiple batches are used 2) the data is probably heavily biased towards 0 steering angle so most likely validation set would match with testing set if they are very small.
Increasing to 100 batches fixed this discrepancy.  As expected with only 10 images and 100 epochs the model did overfit.

The next step was to increase the training set to 100 and decrease epochs to 20.  The thinking was that this would produce more realistic results, while still run easily on my local machine.

At this point I could spend some time on designing the architecture to introduce nonlinearity.  I still kept it simple with only an additional convolution and max pooling layers.  I knew that a potential pitfall was the lack of balance in the data, but out of curiosity I tried the updated model on the simulator as well.  As expecting it didn't take long for the car to go off road.

At this point I was very comfortable with the general pipeline, so the next task was to introduce some balance to the data, increase the training set, and start adding more complicated architecture and testing on a faster machine.

My strategy for the training data was to use the Udacity data set and augment it with some more training data on the simulator where I would concentrate on adding recovery movements (see the previous section).  I gave a different name to this new training data file and would make only corrections when the car would approach side of the road.  This resulted in good training data for recovery, but since I would wait until the car approached the side before doing any steering, I expected the data corresponding to no steering to be bad, so when reading this data I would ignore data with 0 steering angle.  One complication was that my main machine is not very fast so even with the lowest settings on the simulator it was difficult to generate good data.  This presented a challenging problem to generate good driving from combination of good Udacity dataset and lesser quality generated dataset, but trained with specific goal of recovery from going off road.

With these modifications and still a very basic architecture I newly trained the model with 10 epochs and about 11,000 images.  At this point the car was able to navigate (not perfectly) until the bridge and would crash.  Next I decided to not change anything except to do more training.  This gave a good opportunity to practice with loading weights from the already achieved training.  So I loaded the weights and trained for 1 more epoch still with 11,000.  The results were good.  The car was now able to navigate a full track with mostly very nice driving except for a couple of places where it got too close to the edge of the road, but did recover.  Unfortunately it went off course on the second lap.

It was now time to improve the augmentation and architecture.  First I added a 2d cropping layer to make the training more efficient.  

This gave me a chance to think about combining additional architecture steps with loading the weights from a previous architecture.  The next step was the most time consuming as making even small updates to architecture could result in different behavior.  Finally after much experimentation I was able to achieve low validation loss and the car was able to drive autonomously around the track without leaving the road.

#### Visualizations

As mentioned in the previous sections, one of the concerns with the data set is the balance of the data. In particular, since most of the driving is in straight line, the 0 angles are over-represented.  

To improve the data set I recorded much more training data and purposefully ignored 0 angle data in the new datasets. In addition, the sign of the angles is biased.  I augmented the data by introducing flipped images and reversed steering angles. As such the combination of the dataset had a much better balanced data as shown below.
The data is still a little biased towards 0 angles (much better than before augmentation), but due to lower quality of new simulation data, it helps with smoother driving.

The additional datasets were generated with the simulator.  One, where the car is purposefully driven straight until it gets close to the lane edge and then is recovered and so on, this allows to avoid constantly pushing record button.

When the program is ran in the testing mode the processing of images can be visualized.  Below are examples of some of the images, original and processed, as well as what the angle distribution looks like after augmentation (more spread out than originally).

[data_distr]: ./figure_1.png "Data distribution after augmentation"
[blurred_image]: ./figure_2.png "Blurred image"
[blurred_image_flipped]: ./figure_3.png "Flipped Blurred image"
[cropped_image]: ./figure_4.png "Cropped image in BGR color space"


Data distribution after augmentation:
![alt text][data_distr]

Blurred image:
![alt text][blurred_image]

Flipped Blurred image:
![alt text][blurred_image_flipped]

Cropped image in BGR color space:
![alt text][cropped_image]

After the augmentation the final dataset on which the model was trained and validated consisted of around 50,000 images, which were split 80% for training and 20% for testing and shuffled. It was ran for 30 epochs.  
