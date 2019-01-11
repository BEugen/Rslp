import numpy as np
import os
import cv2
import itertools
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import random
import matplotlib.pyplot as plt
import uuid
import scipy.fftpack
import logging

sess = tf.Session()
K.set_session(sess)

LP_LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
              'B', 'C', 'D', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']
IMG_PATH_ROOT = '/mnt/misk/misk/lplate/nchars'
LP_MAX_LENGHT = 9
PREDICT_DETECT_LEVEL = 0.55
PREDICT_FILTER_LEVEL = 0.7

IMG_CROP = '/mnt/misk/misk/lplate/lp-un-mask/img'
IMG_MASK = '/mnt/misk/misk/lplate/lp-un-mask/mask'


class RecognizeLp(object):
    def __init__(self):
        self.cntf = 0
        self.folder_nn = 'nn/'
        self.nn_detect_lp = 'model-detect-lp'
        self.nn_ocr_lp = 'model-ocr-lp'
        self.nn_filter_lp = 'model-filter-lp'
        self.dlp = self.__get_model_detect_lp()
        self.flp = self.__get_model_filter_lp()
        self.ocrlp = self.__get_model_ocr_lp()
        self.pdl = PREDICT_DETECT_LEVEL
        self.fdl = PREDICT_FILTER_LEVEL
        self.letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                        'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']
        self.images_arr = []
        self.char_position = [25, 54, 81, 108, 135, 162, 180, 202, 227]
        # 0   1   2   3    4     5   6    7     8

    def __images_arr_init(self):
        self.images_arr = []
        for i in range(0, LP_MAX_LENGHT):
            self.images_arr.append([])

    # def __plot_images(self, images, grey):
    #     fig = plt.figure(figsize=(15, 18))
    #     for i in range(min(16, len(images))):
    #         fig.add_subplot(4, 4, i + 1)
    #         if grey:
    #             plt.imshow(images[i], cmap='gray')
    #         else:
    #             plt.imshow(images[i])
    #     plt.show()

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
            out = cv2.resize(out, (256, 64))
            # out = np.expand_dims(out.T, -1)/255
            img_gepotise.append(out)
        mdmax = np.max(md_arr)
        maxs = np.where((md_arr >= np.uint32(mdmax)) & (md_arr <= mdmax))
        return np.array(img_gepotise)[maxs]

    def __image_clear_border(self, image, radius=3):
        img = image.copy()
        _, cntr, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        (imgrows, imgcols) = image.shape
        cntrs = []
        for idx in np.arange(len(cntr)):
            cnt = cntr[idx]
            for pt in cnt:
                rowcnt = pt[0][1]
                colcnt = pt[0][0]
                check1 = (0 <= rowcnt < radius) or (imgrows - 1 - radius <= rowcnt < imgrows)
                check2 = (0 <= colcnt < radius) or (imgcols - 1 - radius <= colcnt < imgcols)
                if check1 or check2:
                    cntrs.append(idx)
                    break
        for idx in cntrs:
            cv2.drawContours(img, cntr, idx, (0, 0, 0), -1)
        return img

    def __bw_area_open(self, image, areapixel=10):
        img = image.copy()
        _, cntr, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for idx in np.arange(len(cntr)):
            area = cv2.contourArea(cntr[idx])
            if 0 <= area <= areapixel:
                cv2.drawContours(img, cntr, idx, (0, 0, 0), -1)
        return img

    def __image_denoise(self, image, expandpixel=4, areapixel=20, radius=3, gamma1=0.2, gamma2=1.5, sigma=10,
                        levelthreh=50):
        image = cv2.equalizeHist(image)
        (rows, cols) = image.shape
        img_expand = np.full((rows + expandpixel * 2, cols), 255)
        img_expand[expandpixel:rows + expandpixel, :] = image[:, :]
        (rows, cols) = img_expand.shape
        imglogr = np.log1p(np.array(img_expand, dtype='float') / 255)
        # Create Gaussian mask of sigma 10
        M = 2 * rows + 1
        N = 2 * cols + 1
        (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
        centerx = np.ceil(N / 2)
        centery = np.ceil(M / 2)
        gauss_numerator = (X - centerx) ** 2 + (Y - centery) ** 2
        # Low pass and high pass filter
        hlow = np.exp(-gauss_numerator / (2 * sigma * sigma))
        hhigh = 1 - hlow
        hlowshift = scipy.fftpack.ifftshift(hlow.copy())
        hhighshift = scipy.fftpack.ifftshift(hhigh.copy())

        # filter image and crop
        imagefl = scipy.fftpack.fft2(imglogr.copy(), (M, N))
        imageoutlow = scipy.real(scipy.fftpack.ifft2(imagefl.copy() * hlowshift, (M, N)))
        imageouthigh = scipy.real(scipy.fftpack.ifft2(imagefl.copy() * hhighshift, (M, N)))
        imageout = gamma1 * imageoutlow[0:rows, 0:cols] + gamma2 * imageouthigh[0:rows, 0:cols]

        # anti-log rescale to [0, 1]
        imagehmf = np.expm1(imageout)
        imagehmf = (imagehmf - np.min(imagehmf)) / (np.max(imagehmf) - np.min(imagehmf))
        imagehmf2 = np.array(255 * imagehmf, dtype='uint8')
        imagetreshold = imagehmf2 < levelthreh
        imagetreshold = 255 * imagetreshold.astype('uint8')
        img = self.__image_clear_border(imagetreshold, radius)
        img = self.__bw_area_open(img, areapixel)
        return img

    def __get_split_mask(self, image, lp_number, char_size_min=20, char_size=20, lfirstindex=30, areapixel=5):
        try:
            img = self.__bw_area_open(np.uint8(image.copy()), areapixel)
            mean_imgs = np.mean(img, axis=0)
            fig = plt.figure(figsize=(15, 18))
            imgplot = plt.imshow(img, cmap='gray')
            plt.plot(mean_imgs)
            mask_index_split = []
            index = np.where(mean_imgs >= lfirstindex)
            i = int(np.min(index) - char_size / 2)
            if i < 0:
                i = 4
            mask_index_split.append(i)
            while (i + char_size) < len(mean_imgs):
                index = np.where(mean_imgs[i:i + char_size] == np.min(mean_imgs[i:i + char_size]))
                if len(index[0]) == 0:
                    i += char_size
                    continue
                n = int(np.mean(index)) + i
                if len(mask_index_split) and (n - mask_index_split[len(mask_index_split) - 1]) <= char_size_min:
                    i += char_size
                else:
                    mask_index_split.append(n)
                    i = n + int(char_size / 2)
            for i in range(1, len(mask_index_split)):
                if (mask_index_split[i] - mask_index_split[i - 1]) > char_size * 2.2:
                    mask_index_split.insert(i, int((mask_index_split[i] + mask_index_split[i - 1]) / 2))
            y = np.full((len(mask_index_split),), 20.0)
            plt.scatter(mask_index_split, y, c='red', s=40)
            plt.show()
            idx = -1
            print(mask_index_split)
            for i in range(1, len(mask_index_split)):
                for y in range(0, len(self.char_position)):
                    if mask_index_split[i - 1] < self.char_position[y] < mask_index_split[i]:
                        idx = y
                        break
                out = self.__image_crop(img[:, mask_index_split[i - 1]: mask_index_split[i]])
                out = self.__image_normalisation(out)
                if idx >= 0:
                    self.images_arr[idx].append(out)
                    idx = -1
                plt.imshow(out, cmap='gray')
                plt.show()
            return True
        except:
            logging.exception('')
            return False

    def __image_crop(self, image):
        ret, img = cv2.threshold(image.copy(), 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (_, contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)
        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = np.reshape(approx, (approx.shape[0], 2))
        min_x, min_y = np.min(approx, axis=0)
        max_x, max_y = np.max(approx, axis=0)
        out = np.zeros_like(image)
        cv2.fillPoly(out, pts=[cnt], color=255)
        out[out == 255] = image[out == 255]
        img = np.zeros((max_y - min_y, max_x - min_x))
        img[:img.shape[0], :img.shape[1]] = out[min_y:max_y, min_x:max_x]
        return img

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
            i = int(len(hwi) * 0.5)
            hw = hwi[i: i + 1]
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
        # print("Ma={0}, Mb={1}".format(len(mean_a), len(mean_b)))
        index = np.where((mean_a <= np.min(mean_a)) & (mean_a <= level))
        index = 0 if len(index[0]) == 0 else np.max(index)
        l_bot = index if index > 0 else 0
        index = np.where((mean_b <= np.min(mean_b)) & (mean_b <= level))
        index = hw if len(index[0]) == 0 else np.min(index) + fb
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
            y = 45
            kf = y / image.shape[0]
            x = int(image.shape[1] * kf)
            (ys, xs) = (64, 64)
            image = cv2.resize(image, (x, y))
            imr = np.zeros((ys, xs))
            yo = int(0.5 * ys - image.shape[0] * 0.5)
            xo = int(0.5 * xs - image.shape[1] * 0.5)
            imr[yo:image.shape[0] + yo, xo:image.shape[1] + xo] = image[0:image.shape[0], 0:image.shape[1]]
            return imr
        except:
            logging.exception('')
            return None

    def __image_conversion(self, imglp, lp_number):
        try:
            print(lp_number)
            imglp = np.squeeze(imglp, -1)
            # igm = imglp.copy()
            # igm = cv2.resize(igm, (286, 64))
            # file_name = str(uuid.uuid4()) + '.jpg'
            # cv2.imwrite(os.path.join(IMG_CROP, file_name), igm)
            # igm = self.__image_denoise(igm, areapixel=20, radius=5)
            # cv2.imwrite(os.path.join(IMG_MASK, file_name), igm)
            # imglp = self.__image_denoise(imglp)

            # plt.imshow(imglp, cmap='gray')
            # plt.show()
            result = self.__get_split_mask(imglp, lp_number)
            return result
        except:
            logging.exception('')
            return False

    def __image_rotate(self, img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=255)
        return result

    def __image_ocr(self, images):
        try:
            images = np.array(images) / 255
            images = np.reshape(images, images.shape + (1,))
            predict = self.ocrlp.predict_classes(images)
            return predict
        except:
            logging.exception('')
            return None

    def __image_filter(self, images):
        try:
            images = np.array(images) / 255
            images = np.reshape(images, images.shape + (1,))
            predict = self.flp.predict(images)
            out = []
            for img in predict:
                img_out = np.zeros(img.shape)
                img_out[img >= self.fdl] = 255
                out.append(img_out)
            return out
        except:
            logging.exception('')
            return None

    def recognize(self, image, file):
        self.__images_arr_init()
        img = self.__detect_lp(image, file)
        if img is not None:
            img = self.__image_filter(img)
            for im in img:
                self.__image_conversion(im, file)
        letters_class = []
        # for imgs in self.images_arr:
        # letters_class.append(self.__image_ocr(imgs))
        # for img in imgs:
        #    cv2.imwrite(os.path.join(IMG_PATH_ROOT, str(uuid.uuid4()) + '.jpg'), img)
        number = ''
        for letters in letters_class:
            if letters is None:
                number += ' '
                continue
            ch = np.bincount(letters).argmax()
            number += self.letters[ch]
        print(number)

    # load model for get license plate from image
    def __get_model_detect_lp(self):
        json_file = open(self.folder_nn + self.nn_detect_lp + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.nn_detect_lp + '.h5')
        loaded_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

    # load model for filtration image license plate
    def __get_model_filter_lp(self):
        json_file = open(self.folder_nn + self.nn_filter_lp + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.nn_filter_lp + '.h5')
        loaded_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

    # load model for ocr license plate chars
    def __get_model_ocr_lp(self):
        json_file = open(self.folder_nn + self.nn_ocr_lp + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.nn_ocr_lp + '.h5')
        loaded_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')
        return loaded_model
