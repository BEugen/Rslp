import numpy as np
import cv2
import os
import random
import math
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import img_to_array
from keras import callbacks




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


def img_generator(noise_path, img_path, batch_size):
    list_file = os.listdir(noise_path)
    random.seed(40)
    random.shuffle(list_file)
    ln = len(list_file)
    while True:
        batch_st = 0
        batch_end = batch_size
        while batch_st < ln:
            lim = min(batch_end, ln)
            x_data = []
            y_data = []
            for file in list_file[batch_st:lim]:
                im_n = cv2.resize(cv2.imread(os.path.join(noise_path, file), cv2.IMREAD_GRAYSCALE), (152, 32))
                file = os.path.splitext(file)[0]
                file = file.split('_')
                im = cv2.resize(cv2.imread(os.path.join(img_path, file[len(file) - 1] + '.png'),
                                           cv2.IMREAD_GRAYSCALE), (152, 32))
                if im is None or im_n is None:
                    continue
                sh = im.shape
                if sh[0] == 0 or sh[1] == 0:
                    continue
                #im = cv2.resize(im, (152, 34))
                #im_n = cv2.resize(im_n, (152, 34))
                x_data.append(im.T)
                y_data.append(im_n.T)

            x_data = np.array(x_data, dtype='float') / 255.0
            y_data = np.array(y_data, dtype='float') / 255.0
            yield (x_data.reshape(x_data.shape + (1,)), y_data.reshape(y_data.shape + (1,)))
            batch_st += batch_size
            batch_end += batch_size


def img_steps(path, batch_size):
    return math.floor(len(os.listdir(path)) / batch_size)

def main():
    IMG_PATH_TRAIN = 'X:/Books/FullPlate/data/ddenoise/train/'
    IMG_PATH_TEST = 'X:/Books/FullPlate/data/ddenoise/train/'
    IN_IMG_PATH_TRAIN = 'X:/Books/FullPlate/data/ddenoise/train_in/'
    IN_IMG_PATH_TEST = 'X:/Books/FullPlate/data/ddenoise/train_in/'
    BATCH_SIZE = 128
    EPOSH = 500
    model = autoencoder()
    model.compile(loss='mean_squared_error', optimizer= RMSprop())
    model.summary()
    log = callbacks.CSVLogger('result/log.csv')
    tb = callbacks.TensorBoard(log_dir='result/tensorboard-logs',
                             batch_size=BATCH_SIZE)
    checkpoint = callbacks.ModelCheckpoint('result/weights-nn_denoise{epoch:02d}.h5', monitor='val_loss',
                                               save_best_only=True, save_weights_only=True, verbose=1)
    history = model.fit_generator(img_generator(IN_IMG_PATH_TRAIN, IMG_PATH_TRAIN, BATCH_SIZE),
                                  steps_per_epoch=img_steps(IMG_PATH_TRAIN, BATCH_SIZE), epochs=EPOSH, verbose=1,
                        validation_steps=img_steps(IMG_PATH_TEST, BATCH_SIZE),
                                  validation_data=img_generator(IN_IMG_PATH_TEST, IMG_PATH_TEST, BATCH_SIZE),
                        callbacks=[log, tb, checkpoint])
    score = model.evaluate_generator(img_generator(IN_IMG_PATH_TEST, IMG_PATH_TEST, BATCH_SIZE),
                                     steps=img_steps(IMG_PATH_TEST, BATCH_SIZE))
    print(score)
    model_json = model.to_json()
    with open("model_nn_denoise.json", "w") as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    main()

