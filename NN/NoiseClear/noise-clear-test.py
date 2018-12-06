from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import random
import math
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop


def autoencoder():
    #encoder
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(152, 32, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    #decoder
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    return model


def main():
    model = autoencoder()
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    model.load_weights('result/weights-nn_denoise56.h5')

    FILE_FOLDER = 'X:/Books/FullPlate/data/ddenoise/test_in'
    FILE_FOLDER_S = 'X:/Books/FullPlate/data/ddenoise/den-result'
    files = os.listdir(FILE_FOLDER)
    for file in files:
        im_b = cv2.resize(cv2.imread(FILE_FOLDER + '/' + file, cv2.IMREAD_GRAYSCALE), (152, 32))
        im_b = im_b.T / 255
        im_b = np.expand_dims(im_b, axis=0)
        im_b = np.expand_dims(im_b, axis=-1)
        imp = model.predict(im_b)
        imp = cv2.multiply(imp, 255)
        imp = np.squeeze(imp, axis=0)
        imp = np.squeeze(imp, axis=-1)
        cv2.imwrite(FILE_FOLDER_S + '/' + os.path.splitext(file)[0] + '.jpg',
                    imp.T, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(file)


if __name__ == '__main__':
    main()

