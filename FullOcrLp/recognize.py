import numpy as np
import os
import cv2
import itertools
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sess = tf.Session()
K.set_session(sess)

LP_LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
              'B', 'C', 'D', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']
LP_MAX_LENGHT = 9
PREDICT_DETECT_LEVEL = 0.7


class RecognizeLp(object):
    def __init__(self):
        self.cntf = 0
        self.folder_nn = 'nn/'
        self.nn_detect_lp = 'model-detect-lp'
        self.nn_ocr_lp = 'model-ocr-lp'
        self.dlp = self.__get_model_detect_lp()
        self.ocrlp = self.__get_model_ocr_lp()
        self.pdl = 0.7
        self.letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                        'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']


    def __plot_images(self, images, grey):
        fig = plt.figure(figsize=(15, 18))
        for i in range(min(16, len(images))):
            fig.add_subplot(4, 4, i + 1)
            if grey:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
        plt.show()

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
            out = self.__image_rotate(out, i)
            md = np.median(np.mean(out, axis=1))
            md_arr.append(md)
            out = img[min_y:max_y + 1, min_x:max_x + 1]
            out = self.__image_rotate(out, i)
            #out = cv2.resize(out, (128, 64))
            # out = np.expand_dims(out.T, -1)/255
            img_gepotise.append(out)
        mdmax = np.max(md_arr)
        maxs = np.where((md_arr >= np.uint32(mdmax)) & (md_arr <= mdmax))
        return np.array(img_gepotise)[maxs]

    def __get_split_mask(self, image, lp_number, char_size_min=10, char_size=10):
        imgb = cv2.GaussianBlur(image, (9, 9), 100)
        img = cv2.subtract(imgb, image)
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.GaussianBlur(img, (1, 1), 10)
        ret, img = cv2.threshold(img, 5, 255, 0)
        img = cv2.resize(img, (128, 64))
        mean_imgs = np.mean(img, axis=0)
        mask_index_split = []
        i = 0
        fi = int(char_size / 2)
        index = np.where(mean_imgs[i:i + fi] == np.min(mean_imgs[i:i + fi]))
        i = np.max(index)
        mask_index_split.append(i)
        while (i + char_size) < len(mean_imgs):
            index = np.where(mean_imgs[i:i + char_size] == np.min(mean_imgs[i:i + char_size]))
            n = np.min(index) + i
            print(i, n, n - i)
            if len(mask_index_split) and (n - mask_index_split[len(mask_index_split) - 1]) <= char_size_min:
                i += char_size
            else:
                mask_index_split.append(n)
                i = n + int(char_size / 2)
        images = []
        for i in range(1, len(mask_index_split)):
            images.append(img[:, mask_index_split[i - 1]: mask_index_split[i]])
        return images

    def __img_crop_next(self, img, level_blank=5, level_find=100, axis=0, lt=True):
        mean_imgs = np.mean(img, axis=axis)
        hwi = np.where(mean_imgs >= level_find) if lt else np.where(mean_imgs <= level_find)
        index = np.where(mean_imgs <= level_blank) if lt else np.where(mean_imgs >= level_blank)
        if len(hwi) == 0:
            if axis == 0:
                hw = img.shape[1] * 0.5
            else:
                hw = img.shape[0] * 0.5
        else:
            hwi = hwi[0] if len(hwi) > 0 else []
            i = int(len(hwi)*0.5)
            hw = hwi[i: i+1]
        shape_img = img.shape[1] if axis == 0 else img.shape[0]
        index = index[0] if len(index) > 0 else []
        sl = index[index > hw]
        l_top = np.min(sl) - 1 if len(sl) > 0 else shape_img
        l_top = l_top if l_top >= 0 else shape_img
        sl = index[index < hw]
        l_bot = np.max(sl) + 1 if len(sl) > 0 else 0
        l_bot = l_bot if l_bot >= 0 else 0
        w = img.shape[1] if axis > 0 else l_top - l_bot
        h = img.shape[0] if axis == 0 else l_top - l_bot
        if h < 0:
            h = img.shape[0]
        if w < 0:
            w = img.shape[1]
        imc = np.zeros((h, w))
        imc[:, :] = img[l_bot:l_top, :] if axis > 0 else img[:, l_bot:l_top]
        return imc

    def __img_crop_next_2(self, img, axis=0, level=5, findk=0.15):
        mean_imgs = np.mean(img, axis=axis)
        hw = (img.shape[0] if axis else img.shape[1])
        fa = int(hw * 0.5 - hw * 0.5 * findk)
        fb = int(hw * 0.5 + hw * 0.5 * findk)
        fa = 0 if fa < 0 else fa
        fb = len(mean_imgs) if fb > len(mean_imgs) else fb
        mean_a = mean_imgs[:fa]
        mean_b = mean_imgs[fb:]
        mni = np.median(mean_imgs)
        wsa, ima = self.__max_window(mean_a, mni, 0)
        wsb, imb = self.__max_window(mean_b, mni, fb)
        fa = ima
        fb = imb
        if wsa == 0:
            fa = imb
        if wsb == 0:
            fb = ima
        if wsa * 2.5 < wsb:
            fa = imb
        if wsb * 2.5 < wsa:
            fb = fa
        fa = 0 if fa < 0 else fa
        fb = len(mean_imgs) if fb > len(mean_imgs) else fb
        mean_a = mean_imgs[:fa]
        mean_b = mean_imgs[fb:]
        index = np.where((mean_a <= np.min(mean_a)) & (mean_a <= level))
        index = 0 if len(index[0]) == 0 else np.max(index)
        print(index)
        l_bot = index if index > 0 else 0
        index = np.where((mean_b <= np.min(mean_b)) & (mean_b <= level))
        index = hw if len(index[0]) == 0 else np.min(index) + fb
        print(index)
        l_top = index if index <= hw else hw
        w = img.shape[1] if axis > 0 else l_top - l_bot
        h = img.shape[0] if axis == 0 else l_top - l_bot
        imc = np.zeros((h, w))
        imc[:, :] = img[l_bot:l_top, :] if axis > 0 else img[:, l_bot:l_top]
        return imc

    def __max_window(self, data, level, offset=0):
        index = np.where(data >= level)
        if len(index) == 0 or len(index[0]) == 0:
            return 0, 0
        __index = index[0]
        windows_size = __index[len(__index) - 1] - __index[0]
        windows_max_value = int(np.mean(index)) + offset
        return windows_size, windows_max_value


    def __image_normalisation(self, image):
        try:
            image = cv2.resize(image, (24, 38))
            imr = np.zeros((64, 64))
            yo = int(0.5 * 64 - image.shape[0] * 0.5)
            xo = int(0.5 * 64 - image.shape[1] * 0.5)
            imr[yo:image.shape[0] + yo, xo:image.shape[1] + xo] = image[0:image.shape[0], 0:image.shape[1]]
            return imr
        except:
            return None

    def __image_conversion(self, imglp, lp_number):
        print(lp_number)
        images = self.__get_split_mask(imglp, lp_number)
        kernel = np.ones((5, 1), np.uint8)
        for i in range(0, len(images)):
            img = cv2.erode(images[i].copy(), kernel, iterations=1)
            (_, contours, _) = cv2.findContours(np.uint8(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if len(contours) == 0:
                continue
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
            out[min_y:max_y, min_x:max_x] = 255
            out[out == 255] = images[i][out == 255]
            ret, out = cv2.threshold(out, 100, 255, 0)
            out = self.__img_crop_next_2(out, axis=1)
            out = self.__img_crop_next_2(out, axis=0)
            images[i] = out
            # images[i] = img_crop_next_2(images[i], axis=1)
            # images[i] = img_crop_next_2(images[i], axis=0)

        return images

    def __image_rotate(self, img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __image_ocr(self, images):
        images = np.array(images)/255
        images = np.reshape(images, images.shape + (1, ))
        predict = self.ocrlp.predict_classes(images)
        return predict

    def recognize(self, image, file):
        img = self.__detect_lp(image, file)
        if img is not None:
            for im in img:
                images = self.__image_conversion(im, file)
                lps = self.__image_ocr(images)
                number = ''
                for ch in lps:
                    number += self.letters[ch]
                print(number)
        else:
            print('bad!!!')

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

    def __get_model_detect_lp(self):
        json_file = open(self.folder_nn + self.nn_detect_lp + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.nn_detect_lp + '.h5')
        loaded_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model


    def __get_model_ocr_lp(self):
        json_file = open(self.folder_nn + self.nn_ocr_lp + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.nn_ocr_lp + '.h5')
        loaded_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')
        return loaded_model

    def _random_file_name(self):
        file_name = ''
        for i in range(0, 12):
            random.seed()
            file_name += LP_LETTERS[random.randint(0, len(LP_LETTERS)-1)]
        return file_name

