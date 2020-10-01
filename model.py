import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imresize

def next_image_set(batch_size=64):
    # Method that creates set of images and measurements given size
    data = pd.read_csv("./data/driving_log.csv")
    num_of_img = len(data)
    random_indices = np.random.randint(0, num_of_img, batch_size)

    images_and_measurements = []
    for i in random_indices:
        image_choise = np.random.randint(0, 3)
        if image_choise == 0:
            image = data.iloc[i]['center'].strip()
            measurement = data.iloc[i][3]
            images_and_measurements.append((image, measurement))
        elif image_choise == 1:
            image = data.iloc[i]['left'].strip()
            measurement = data.iloc[i][3] + 0.2
            images_and_measurements.append((image, measurement))
        else:
            image = data.iloc[i]['right'].strip()
            measurement = data.iloc[i][3] - 0.2
            images_and_measurements.append((image, measurement))

    return images_and_measurements

def process_image(image, measurement):
    p = np.random.randint(0, 20)
    if p % 2 == 0:
        image = np.fliplr(image)
        measurement = -measurement
    
    return image, measurement

def data_generator(batch_size=64):
    # generator method that yields data batch
    while True:
        X_batch = []
        y_batch = []
        images_and_measurements = next_image_set(batch_size)
        for image, measurement in images_and_measurements:
            raw_image = cv2.imread("./data/" + image)
            raw_measurement = measurement
            new_image, new_measurement = process_image(raw_image, raw_measurement)
            X_batch.append(new_image)
            y_batch.append(new_measurement)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)


# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, Cropping2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding="valid"))
model.add(Activation('relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="valid"))
model.add(Activation('relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="valid"))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')

X_train = data_generator()
X_valid = data_generator()

model.fit_generator(X_train, validation_data=X_valid, verbose=2, steps_per_epoch=300, epochs=5, validation_steps=640)

model.save('model.h5')

model.summary()

