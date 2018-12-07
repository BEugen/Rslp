import numpy as np
import os
import cv2
import itertools
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import zlib
import math

sess = tf.Session()
K.set_session(sess)

LP_LETTERS = {'0', '1,' '2', '3', '4', '5', '6', '7', '8', '9', 'A',
              'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' '}
LP_MAX_LENGHT = 9
PREDICT_DETECT_LEVEL = 0.7

class RecognizeLp(object):
    def __init__(self):
        self.folder_nn = 'nn/'
        self.nn_detect_lp = 'model-detect-lp'
        self.nn_ocr_lp = 'model-ocr-lp'
        self.letters = sorted(list(LP_LETTERS))
        self.max_len = LP_LETTERS
        self.pdl = PREDICT_DETECT_LEVEL
        self.dlp = self.__get_model_detect_lp()
        self.ocrlp = self.__get_model_ocr_lp()

    def __detect_lp(self, img):
        img_crop = cv2.resize(img, (224, 224))
        img_crop = img_crop / 255
        img_crop = np.reshape(img_crop, (1, img_crop.shape[0], img_crop.shape[1], 1))
        pred = self.dlp.predict(img_crop)[0]
        mask = np.zeros(pred.shape)
        mask[pred >= self.pdl] = 255
        mask = cv2.resize(mask, img.shape).T
        mask = mask.astype(np.uint8)
        cv2.imwrite('test.jpg', mask)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if len(contours) == 0:
            return None
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)
        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)
        epsilon = 0.025 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = np.reshape(approx, (approx.shape[0], 2))
        min_x, min_y = np.min(approx, axis=0)
        max_x, max_y = np.max(approx, axis=0)
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]
        img_gepotise = []
        for i in range(-10, 10):
            out = img[min_y:max_y + 1, min_x:max_x + 1]
            out = self.__rotateimage(out, i)
            out = cv2.GaussianBlur(out, (5, 5), 0)
            out = cv2.adaptiveThreshold(out, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 9, 3)
            out = cv2.resize(out, (152, 34)) / 255
            img_gepotise.append(out)
        return img_gepotise


    def __rotateimage(self, img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def __ocr_license_plate(self, imgs):
        imgs = np.expand_dims(imgs, 0)
        net_inp = self.ocrlp.get_layer(name='the_input').input
        net_out = self.ocrlp.get_layer(name='softmax').output
        net_out_value = sess.run(net_out, feed_dict={net_inp: imgs})
        snn1text = self.__decode_batch(net_out_value)

    def recognize(self, image):
        img = self.__detect_lp(image)
        self.__ocr_license_plate(img)

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def __decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(self.letters):
                    outstr += self.letters[c]
            ret.append(outstr)
        return ret

    def __get_model_ocr_lp(self):
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        loaded_model = load_model(self.folder_nn + self.nn_ocr_lp + '.h5', compile=False)
        loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        return loaded_model

    def __get_model_detect_lp(self):
        json_file = open(self.folder_nn + self.nn_detect_lp + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.nn_detect_lp + '.h5')
        loaded_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

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

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model


