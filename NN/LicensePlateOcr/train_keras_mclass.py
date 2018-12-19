from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import cv2, numpy as np
import os
from keras.callbacks import TensorBoard
from time import time
import random
from keras import callbacks
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_PATH_TRAIN = '/mnt/misk/misk/lplate/chars/train/'
IMG_PATH_TEST= '/mnt/misk/misk/lplate/chars/test/'
#IMG_PATH = 'Imgtrain/'
BATCH_SIZE = 64
NB_EPOCH = 100
NB_CLASSES = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.25
INIT_LR = 1e-3
OPTIM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)



def LeNet():
    model = Sequential()
    # CONV => RELU => POOL
    model.add(Conv2D(20, kernel_size=3, padding="same",
                     input_shape=(64, 64, 1), kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    # CONV => RELU => POOL
    model.add(Conv2D(50, kernel_size=5, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(70, kernel_size=3, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(50, kernel_size=5, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(70, kernel_size=3, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.30))
    # Flatten => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # a softmax classifier
    model.add(Dense(22))
    model.add(Activation("softmax"))

    return model


def load_image(path):
    list_file = os.listdir(path)
    random.seed(40)
    random.shuffle(list_file)
    x_data = []
    y_data = []
    for file in list_file:
        flabel = os.path.splitext(file)[0].split('_')
        im = cv2.resize(cv2.imread(path + file,  cv2.IMREAD_GRAYSCALE), (64, 64))
        im = img_to_array(im)
        x_data.append(im)
        y_data.append(flabel[len(flabel)-1])
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data)
    return x_data, y_data


def main(args):
    X_train, Y_train = load_image(IMG_PATH_TRAIN)
    Y_train = np_utils.to_categorical(Y_train, num_classes=22)
    X_test, Y_test = load_image(IMG_PATH_TEST)
    Y_test= np_utils.to_categorical(Y_test, num_classes=22)
    #(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=.25, random_state=40)
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}_lp_ocr.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True, write_grads=True, write_images=True,
     #                         histogram_freq=1)
    # fit
    model = LeNet()
    model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=VERBOSE,
                        validation_data=(X_test, Y_test),
                        validation_split=VALIDATION_SPLIT, callbacks=[log, tb, checkpoint])

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('Test score:', score[0])
    print('Test accuracy', score[1])

    # save model
    model_json = model.to_json()
    with open("model_lp_ocr.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_lp_ocr.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--save_dir', default='./result-lp-ocr')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)
