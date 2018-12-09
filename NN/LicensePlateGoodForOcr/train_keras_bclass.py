from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
import cv2
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import random
from keras import callbacks
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_PATH_TRAIN = '/mnt/misk/misk/lplate/data/data_rt/train/'
IMG_PATH_TEST = '/mnt/misk/misk/lplate/data/data_rt/test/'
BATCH_SIZE = 32
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
                     input_shape=(224, 224, 1), kernel_initializer='he_normal'))
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
    model.add(Dense(1))
    model.add(Activation("softmax"))

    return model

def Unet(size):
    inputs = Input(size)

    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ELU()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ELU()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ELU()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ELU()(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ELU()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ELU()(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ELU()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ELU()(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ELU()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ELU()(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ELU()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ELU()(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ELU()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ELU()(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ELU()(conv9)

    conv10 = Conv2D(16, (3, 3), activation='sigmoid')(conv9)
    flat1 = Flatten()(conv10)
    dense1 = Dense(500, activation='relu')(flat1)
    dense2 = Dense(1, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=dense2)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_image(path):
    list_file = os.listdir(path)
    random.seed(40)
    random.shuffle(list_file)
    x_data = []
    y_data = []
    for file in list_file:
        flabel = os.path.splitext(file)[0].split('_')
        im = cv2.imread(path + file,  cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (128, 128))
        im = img_to_array(im)
        x_data.append(im)
        y_data.append(flabel[len(flabel)-1])
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data, dtype='int')
    return x_data, y_data


def main(args):
    X_train, Y_train = load_image(IMG_PATH_TRAIN)
    #Y_train = np_utils.to_categorical(Y_train, num_classes=1)
    X_test, Y_test = load_image(IMG_PATH_TEST)
    #Y_test= np_utils.to_categorical(Y_test, num_classes=1)
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}-lp-gd.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    model = Unet((128, 128, 1))
    print(model.summary())
    #model.compile(loss='binary_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=VERBOSE,
                        validation_data=(X_test, Y_test),
                        validation_split=VALIDATION_SPLIT, callbacks=[log, tb, checkpoint])

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('Test score:', score[0])
    print('Test accuracy', score[1])

    # save model
    model_json = model.to_json()
    with open("model-lp-gd.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model-lp-gd.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--save_dir', default='./result-fnn')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)
