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

    def __detect_lp(self, img, file):
        img_crop = cv2.resize(img, (224, 224))
        img_crop = img_crop / 255
        img_crop = np.reshape(img_crop, (1, img_crop.shape[0], img_crop.shape[1], 1))
        pred = self.dlp.predict(img_crop)[0]
        mask = np.zeros(pred.shape)
        mask[pred >= self.pdl] = 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = mask.astype(np.uint8)
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
        md_arr = []
        for i in range(-10, 10):
            out = mask[min_y:max_y + 1, min_x:max_x + 1]
            out = self.__rotateimage(out, i)
            md = np.median(np.mean(out, axis=0))
            #cv2.imwrite(str(i + 10) + '_' + str(math.floor(md)) + '_test.jpg', out)
            md_arr.append(md)
            out = img[min_y:max_y + 1, min_x:max_x + 1]
            out = self.__rotateimage(out, i)
            out = cv2.resize(out, (152, 34))
           # out = np.expand_dims(out.T, -1)/255
            img_gepotise.append(out)
        mdmax = np.max(md_arr)
        maxs = np.where((md_arr >= round(mdmax, 0)) & (md_arr <= mdmax))

        return np.array(img_gepotise)[maxs]


    def __clip_chars(self, images):
        for img in images:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 9, 3)
            img[img.shape[0] - 4: img.shape[0], 0:img.shape[1]] = 255
            img[0: 4, 0:img.shape[1]] = 255
            mn = []
            for i in range(0, img.shape[1]):
                mn.append(np.mean(img[:, i:i + 3]))
                i += 4
            md = np.mean(mn)
            maxs = np.where(mn < md*0.8)
            return
            ret, img = cv2.threshold(img, 100, 255, 0)
            im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            for cnt in contours:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx = np.reshape(approx, (approx.shape[0], 2))
                min_x, min_y = np.min(approx, axis=0)
                max_x, max_y = np.max(approx, axis=0)
                if (max_x - min_x) > 0:
                    koeff = math.fabs((max_y - min_y) / (max_x - min_x))
                    if koeff <= 0.3 and cv2.contourArea(cnt) < 500.0:
                        print(koeff, cv2.contourArea(cnt))
                        cv2.fillPoly(img, pts=[cnt], color=(255, 255, 255))
            cv2.imshow("Contour-r", cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))
            cv2.waitKey(2000)
            im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for cnt in contours:
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx = np.reshape(approx, (approx.shape[0], 2))
                min_x, min_y = np.min(approx, axis=0)
                max_x, max_y = np.max(approx, axis=0)
                if (max_x - min_x) > 0:
                    koeff = math.fabs((max_y - min_y) / (max_x - min_x))
                    if 0.5 < koeff < 2.2 and cv2.contourArea(cnt) > 80:
                        print(koeff, max_x - min_x)
                        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
                # cv2.drawContours(canvas, approx, -1, (0, 255, 0), 1)
            cv2.imshow("Contour", cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))
            cv2.waitKey(2000)


    def __rotateimage(self, img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def __ocr_license_plate(self, imgs):
        net_inp = self.ocrlp.get_layer(name='the_input').input
        net_out = self.ocrlp.get_layer(name='softmax').output
        net_out_value = sess.run(net_out, feed_dict={net_inp: imgs})
        lptext = self.__decode_batch(net_out_value)
        print(lptext)

    def recognize(self, image, file):
        img = self.__detect_lp(image, file)
        if img is not None:
            self.__clip_chars(img)
            #self.__ocr_license_plate(img)
        else:
            print('bad!!!')

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



