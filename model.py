import pickle
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l1,l2
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2, random
import seaborn as sns

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('testing', False, "Testing flag") # will show a random processed image if true
flags.DEFINE_integer('epochs', 50, "The number of epochs.")

# some image processing for the files used for training
# to help with generalization - images are darkened and blurred
def process_image(image, training=True):
    if training:
        image_br = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_br = cv2.GaussianBlur(image_br, (3,3),0)
        image_br = np.round(image_br * 0.98)
    else:
        image_br = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_br

# data is loaded from three datasets
# - driving_log.csv is the Udacity dataset
# - driving_left_log.csv is a dataset generated with the simulator where the car
# is purposefully driven straight until it gets closed to the lane edge and
# then is recovered and so on, this allows to avoid constantly pushing record
# the problem is that the straight driving data is bad driving, so only nonzero
# angles are read, this also balances the data.  This data is left biased, but is
# flipped as part of the training
# - driving_log_recovery.csv is an additional data set generated from the simulator
# now with constant good driving so the zero angles are included
def load_data(testing=False):

    import csv
    path = 'data/'
    car_images = []
    steering_angles = []

    file_row_array = []
    angles = np.array([])
    with open('data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            angle = float(row[3].strip())
            file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
            angles = np.append(angles, angle)

    # read data from left biased
    with open('data/driving_left_log.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            angle = float(row[3].strip())
            if angle != 0:
                file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
                angles = np.append(angles, angle)

    # read data from recovery
    with open('data/driving_log_recovery.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            angle = float(row[3].strip())
            file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
            angles = np.append(angles, angle)

    # visualize dataset to check for well balanced data
    if testing:
        plt.figure()
        sns.distplot(angles)

    return file_row_array

# generator that loads data in batches to avoid memory problems
# this is also where the data is augmented
# if the training parameter is true the images are also blurred and darkened
# in process_image(..)
# also left and right cameras are used with steering adjustment
# and the images are flipped to eliminate left-bias
def generator(samples, training, batch_size=32):
    num_samples = len(samples)
    if training:
        # adjust batch since since 6 times more data is generated through
        # flipping and using left and right cameras
        batch_size = batch_size // 6

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                img_center = process_image(cv2.imread(name), training)
                img_flipped = np.fliplr(img_center)
                steering_center = float(batch_sample[3])
                car_images.extend([img_center, img_flipped])
                steering_angles.extend([steering_center, -steering_center])
                if training:
                    # create adjusted steering measurements for the side camera images
                    correction = 0.15 # this is a parameter to tune

                    # use additional cameras and
                    # generated flipped images to avoid one-sided bias
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    img_left = process_image(cv2.imread(name), training)
                    img_left_flipped = np.fliplr(img_left)
                    steering_left = float(batch_sample[3]) + correction

                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    img_right = process_image(cv2.imread(name), training)
                    img_right_flipped = np.fliplr(img_right)
                    steering_right = float(batch_sample[3]) - correction

                    # add images and angles to data set
                    car_images.extend([img_left, img_left_flipped, img_right, img_right_flipped])
                    steering_angles.extend([steering_left, -steering_left, steering_right, -steering_right])

            X_train = np.array(car_images)
            y_train = np.array(steering_angles)

            yield shuffle(X_train, y_train)


def main(_):
    testing = FLAGS.testing
    row_data = load_data(testing)
    # split data
    train_samples, validation_samples = train_test_split(row_data, train_size=0.8)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, training=True, batch_size=33)
    validation_generator = generator(validation_samples, training=False, batch_size=32)

    model = Sequential()
    model.add(Cropping2D(cropping=((40,15), (0,0)), input_shape=(160,320,3)))

    # Visualize the cropping for testing purposes
    if testing:
        layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
        input_image = cv2.imread('./data/IMG/'+train_samples[0][0].split('/')[-1])
        input_image = np.uint8(process_image(input_image))
        cropped_image = layer_output([input_image[None,...]])[0][0,...]
        cropeed_image = np.uint8(cropped_image)
        flipped_image = np.fliplr(input_image)
        flipped_image = np.uint8(flipped_image)
        plt.figure()
        plt.imshow(input_image)
        plt.figure()
        plt.imshow(flipped_image)
        plt.figure()
        plt.imshow(cropped_image)
        plt.show()

    # construct the model - l2 regularization is used to help generalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(2, 2, 2, W_regularizer=l1(0.000001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Convolution2D(4, 2, 2, W_regularizer=l1(0.000001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(32, name="dense2"))
    model.add(Activation('elu'))
    model.add(Dense(1, name="dense1"))

    if not testing:
        model.compile(optimizer=Adam(lr=0.0001), loss="mse")
        checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
        history_object = model.fit_generator(generator = train_generator, samples_per_epoch =
            6*len(train_samples), validation_data =
            validation_generator,
            nb_val_samples = len(validation_samples),
            nb_epoch=FLAGS.epochs, verbose=1, callbacks=[checkpoint])
        model.save('model.h5')

        # performance visualization
        print(history_object.history.keys())

        ### plot the training and validation loss for each epoch
        plt.figure()
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
