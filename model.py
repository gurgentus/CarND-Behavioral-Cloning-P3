import pickle
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2, random
import pandas as pd
import seaborn as sns
from keras import backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('testing', False, "Testing flag")
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")

def process_image(image):
    d = random.randint(-3, 3)
    image = image * (1-d/10)
    return image

def load_data():

    import csv
    path = 'data/'
    car_images = []
    steering_angles = []


    with open('data/driving_log.csv', 'r') as f:
        reader = csv.reader(f) #, delimiter =', ')
        next(reader)
        count = 0
        file_row_array = []
        angles = np.array([])
        zeroCount = 0
        for row in reader:

            angle = float(row[3].strip())
            if angle == 0:
                zeroCount = zeroCount + 1

            if angle != 0 or zeroCount == 5:
                file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
                angles = np.append(angles, angle)

            if zeroCount == 5:
                zeroCount = 0

    # read data from left biased
    with open('data/driving_left_log.csv', 'r') as f:
        reader = csv.reader(f) #, delimiter =', ')
        next(reader)
        count = 0
        for row in reader:
            angle = float(row[3].strip())
            if angle != 0:
                file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
                angles = np.append(angles, angle)

    # read data from right biased
    with open('data/driving_right_log.csv', 'r') as f:
        reader = csv.reader(f) #, delimiter =', ')
        next(reader)
        count = 0
        for row in reader:
            angle = float(row[3].strip())
            if angle != 0:
                file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
                angles = np.append(angles, angle)

    # read data from jungle
    # read data from right biased
    # with open('data/driving_jungle_log.csv', 'r') as f:
    #     reader = csv.reader(f) #, delimiter =', ')
    #     next(reader)
    #     count = 0
    #     for row in reader:
    #         angle = float(row[3].strip())
    #         file_row_array.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
    #         angles = np.append(angles, angle)


    plt.figure()
    sns.distplot(angles)
    return file_row_array

    # image1 = Image.open('image1.jpg').crop((10, 20, 200, 250)).resize((32, 32), Image.ANTIALIAS)
    # image2 = Image.open('image2.jpg').crop((500, 100, 700, 300)).resize((32, 32), Image.ANTIALIAS)
    # image3 = Image.open('image3.jpg').crop((1800, 500, 3000, 2000)).resize((32, 32), Image.ANTIALIAS)
    # image4 = Image.open('image4.jpg').crop((530, 60, 640, 160)).resize((32, 32), Image.ANTIALIAS)
    # image5 = Image.open('image5.jpg').crop((220, 100, 400, 260)).resize((32, 32), Image.ANTIALIAS)
    #
    #
    # fig = plt.figure()
    # plt.imshow(image1)
    # fig = plt.figure()
    # plt.imshow(image2)
    # fig = plt.figure()
    # plt.imshow(image3)
    # fig = plt.figure()
    # plt.imshow(image4)
    # fig = plt.figure()
    # plt.imshow(image5)
    #
    # arr = [np.array(image1), np.array(image2), np.array(image3), np.array(image4), np.array(image5)]

def generator(samples, extra_cameras, batch_size=2048):
    num_samples = len(samples)
    if extra_cameras:
        batch_size = batch_size // 3

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                img_center = process_image(cv2.imread(name))
                steering_center = float(batch_sample[3])
                car_images.append(img_center)
                steering_angles.append(steering_center)
                if extra_cameras:
                    # create adjusted steering measurements for the side camera images
                    correction = 0.2 # this is a parameter to tune

                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    img_left = process_image(cv2.imread(name))
                    steering_left = float(batch_sample[3]) + correction

                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    img_right = process_image(cv2.imread(name))
                    steering_right = float(batch_sample[3]) - correction

                    # add images and angles to data set
                    car_images.extend([img_left, img_right])
                    steering_angles.extend([steering_left, steering_right])

            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)

            yield shuffle(X_train, y_train)


def main(_):
    row_data = load_data()
    #print(X_train)
    #print(X_train.shape, y_train.shape)
    #print(X_val.shape, y_val.shape)
    row_data = shuffle(row_data)


    train_samples, validation_samples = train_test_split(row_data[0:20000], train_size=0.8)

    print(train_samples)
    print(validation_samples)
    # # test run
    # from keras.datasets import cifar10
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # y_train = y_train.reshape(-1)
    # y_test = y_test.reshape(-1)


    #nb_classes = len(np.unique(y_train))

    #input_shape = X_train.shape[1:]

    # compile and train the model using the generator function
    train_generator = generator(train_samples, extra_cameras=True, batch_size=33)
    validation_generator = generator(validation_samples, extra_cameras=False, batch_size=32)

    #ch, row, col = 3, 80, 320  # Trimmed image format

    # model = Sequential()
    # # Preprocess incoming data, centered around zero with small standard deviation
    # model.add(Lambda(lambda x: x/127.5 - 1.,
    #         input_shape=(ch, row, col),
    #         output_shape=(ch, row, col)))
    # model.add(... finish defining the rest of your model architecture here ...)


    testing = FLAGS.testing
    # Flow 0
    model = Sequential()

    model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))

    # Visualize for testing purposes
    if testing:
        layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
        input_image = cv2.imread('./data/IMG/'+train_samples[0][0].split('/')[-1])
        cropped_image = layer_output([input_image[None,...]])[0][0,...]
        plt.figure()
        plt.imshow(input_image)
        plt.figure()
        plt.imshow(cropped_image)
        plt.show()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5,  input_shape=(160,320,3)))
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(128, name="dense2"))
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1, name="dense1"))


    # Flow 1
    # model = Sequential()
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))

    # # Flow 2
    # inp = Input(shape=input_shape)
    # x = Flatten()(inp)
    # x = Dense(nb_classes, activation='softmax')(x)
    # model = Model(inp, x)

    # Flow 3
    # Build Convolutional Pooling Neural Network with Dropout in Keras Here
    # model = Sequential()
    # model.add(Convolution2D(32, 1, 1, input_shape=input_shape))
    # model.add(MaxPooling2D((1, 1)))
    # model.add(Dropout(0.5))
    # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))

    # if need to pre/postprocess data
    #X_normalized = np.array(X_train / 255.0 - 0.5 )
    #from sklearn.preprocessing import LabelBinarizer
    #label_binarizer = LabelBinarizer()
    #y_one_hot = label_binarizer.fit_transform(y_train)
    #history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)
    #model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])


    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    # model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)


    #history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=2, validation_split=0.2)
    if not testing:
        model.load_weights('model5.h5')
        #weights = model.layers[5].get_weights()

        model.compile(optimizer="adam", loss="mse")
        history_object = model.fit_generator(generator = train_generator, samples_per_epoch =
            3*len(train_samples), validation_data =
            validation_generator,
            nb_val_samples = len(validation_samples),
            nb_epoch=FLAGS.epochs, verbose=1)
        model.save('model.h5')
        ### print the keys contained in the history object
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

    # preprocess data
    #X_normalized_test = np.array(X_test / 255.0 - 0.5 )
    #y_one_hot_test = label_binarizer.fit_transform(y_test)

    # print("Testing")
    #
    # metrics = model.evaluate(X_normalized_test, y_one_hot_test)
    # for metric_i in range(len(model.metrics_names)):
    #     metric_name = model.metrics_names[metric_i]
    #     metric_value = metrics[metric_i]
    #     print('{}: {}'.format(metric_name, metric_value))

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    # TODO: train your model here


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
