import numpy as np
import os
import cv2
import re
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import uuid
import scipy.fftpack
import logging
from datetime import datetime

LP_LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
              'B', 'C', 'D', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']
LP_MAX_LENGHT = 9


class RecognizeLp(object):
    def __init__(self, detect_koeff=0.10, detect_area=250.0, predict_detect_level=0.15, predict_filter_level=0.6,
                 predict_char_level=0.90, image_offset=1):
        self.cntf = 0
        self.folder_nn = 'nn/'
        self.nn_detect_lp = 'model-detect-lp'
        self.nn_ocr_lp = 'model-ocr-lp'
        self.nn_filter_lp = 'model-filter-lp'
        self.dlp = self.__get_model_detect_lp()
        self.flp = self.__get_model_filter_lp()
        self.ocrlp = self.__get_model_ocr_lp()
        self.pdl = predict_detect_level
        self.fdl = predict_filter_level
        self.cdl = predict_char_level
        self.detect_k = detect_koeff
        self.detect_area = detect_area
        self.image_offset = image_offset
        self.detect_number = []
        self.letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                        'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']
        self.number_format = 'cdddccddd'
        self.number_ocr = ''
        self.date_ocr = ''
        self.ok_ocr = False

    def __detect_lp(self, img):
        try:
            img_crop = cv2.GaussianBlur(img.copy(), (5, 5), 1)
            img_crop = cv2.resize(img_crop, (224, 224))
            img_crop = img_crop / 255
            img_crop = np.reshape(img_crop, (1, img_crop.shape[0], img_crop.shape[1], 1))
            pred = self.dlp.predict(img_crop)[0]
            mask = np.zeros(pred.shape)
            mask[pred >= self.pdl] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = mask.astype(np.uint8)
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            result = []
            for cont in contours:
                epsilon = 0.005 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                area = cv2.contourArea(approx)
                print(x, y, h / w if w > 0 else 0, area)
                img_gepotise = []
                md_arr = []
                if w > 0 and h / w >= self.detect_k and area > self.detect_area:
                    for i in range(-10, 10):
                        out = mask[y - self.image_offset:y + h + self.image_offset,
                              x - self.image_offset:x + w + self.image_offset]
                        out = self.__image_rotate(out, i)
                        md = np.median(np.mean(out, axis=1))
                        md_arr.append(md)
                        out = img[y - self.image_offset:y + h + self.image_offset,
                              x - self.image_offset:x + w + self.image_offset]
                        out = self.__image_rotate(out, i)
                        out = cv2.resize(out, (256, 64))
                        img_gepotise.append(out)
                    if len(md_arr) == 0:
                        continue
                    mdmax = np.max(md_arr)
                    maxs = np.where((md_arr >= np.uint32(mdmax)) & (md_arr <= mdmax))
                    images = np.array(img_gepotise)[maxs]
                    img_packet = []
                    for im in images:
                        img_packet.append(im)
                        img_packet.append(self.__image_pre_filter(im))
                        result.append(img_packet)
            return np.array(result)
        except:
            logging.exception('')
            return None

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
        # image = cv2.equalizeHist(image)
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

    def __split_number(self, image, folder, char_size_min=18, areapixel=30, split_level=3):
        try:
            _, tresh = cv2.threshold(np.uint8(image.copy()), 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img = self.__bw_area_open(tresh, areapixel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 40))
            im_morph = cv2.morphologyEx(img.copy(), cv2.MORPH_CLOSE, kernel)
            mean_imgs = np.mean(im_morph, axis=0)
            plt.figure(figsize=(15, 18))
            plt.imshow(img, cmap='gray')
            plt.plot(mean_imgs)
            mask_index_split = []
            index = np.where(mean_imgs > split_level)
            min_lp = np.min(index)
            if min_lp > 2 * char_size_min:
                return []
            min_lp = min_lp - 2 if (min_lp - 2) > 0 else min_lp
            index = np.where(mean_imgs < split_level)
            index = index[0] if len(index) > 0 else []
            index = index[index >= min_lp]
            prev_index = index[0]
            mask_crop = []
            for i in range(1, len(index)):
                if (index[i] - index[i - 1]) > 2 or index[i] == (img.shape[1] - 1):
                    delta = int((index[i - 1] - prev_index) / 2 + prev_index)
                    mask_index_split.append(delta)
                    prev_index = index[i]
            mask_crop.append(mask_index_split[0])
            for i in range(1, len(mask_index_split)):
                if (mask_index_split[i] - mask_index_split[i - 1]) > char_size_min * 2.2:
                    mask_crop.insert(i, int((mask_index_split[i] + mask_index_split[i - 1]) / 2))
                if (mask_index_split[i] - mask_index_split[i - 1]) >= char_size_min:
                    mask_crop.append(mask_index_split[i])
            img_chars = []
            for i in range(1, len(mask_crop)):
                out = self.__image_crop(img[:, mask_crop[i - 1]: mask_crop[i]])
                out = self.__image_normalisation(out)
                img_chars.append(out)
            y = np.full((len(mask_crop),), 20.0)
            plt.scatter(mask_crop, y, c='blue', s=40)
            plt.savefig(os.path.join(folder, str(uuid.uuid4()) + '.png'))
            plt.close()
            return img_chars
        except:
            logging.exception('')
            return []

    def __get_split_mask(self, image, folder, char_size_min=20, char_size=20,
                         lfirstindex=30, areapixel=5, split_level=10):
        try:
            img = self.__bw_area_open(np.uint8(image.copy()), areapixel)
            mean_imgs = np.mean(img, axis=0)
            plt.figure(figsize=(15, 18))
            plt.imshow(img, cmap='gray')
            plt.plot(mean_imgs)
            mask_index_split = []
            index = np.where(mean_imgs >= lfirstindex)
            i = int(np.min(index) - char_size / 2)
            if i < 0:
                i = 4
            mask_index_split.append(i)
            while (i + char_size) < len(mean_imgs):
                index = np.where(mean_imgs[i:i + char_size] <= split_level)
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
            plt.savefig(os.path.join(folder, str(uuid.uuid4()) + '.png'))
            idx = -1
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
                # plt.imshow(out, cmap='gray')
                # plt.show()
            return True
        except:
            logging.exception('')
            return False

    def __image_crop(self, image):
        # ret, img = cv2.threshold(image.copy(), 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ##kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        ##img = cv2.morphologyEx(image.copy(), cv2.MORPH_CLOSE, kernel)
        img = self.__bw_area_open(np.uint8(image.copy()), 5)
        (min_x, min_y, max_x, max_y) = img.shape[1], img.shape[0], 0, 0
        (_, contours, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            epsilon = 0.005 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            min_x = x if x < min_x else min_x
            min_y = y if y < min_y else min_y
            max_x = (x + w) if max_x < (x + w) else max_x
            max_y = (y + h) if max_y < (y + h) else max_y
        #im = np.zeros((max_y - min_y, max_x - min_x))
        im = img[min_y:max_y, min_x:max_x]
        return im

        # (_, contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        # max_area = cv2.contourArea(cnt)
        # for cont in contours:
        #     if cv2.contourArea(cont) > max_area:
        #         cnt = cont
        #         max_area = cv2.contourArea(cont)
        # epsilon = 0.005 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        # approx = np.reshape(approx, (approx.shape[0], 2))
        # min_x, min_y = np.min(approx, axis=0)
        # max_x, max_y = np.max(approx, axis=0)
        # # out = np.zeros_like(image)
        # # cv2.fillPoly(out, pts=[cnt], color=255)
        # # out[out == 255] = image[out == 255]
        # img = np.zeros((max_y - min_y, max_x - min_x))
        # img[:img.shape[0], :img.shape[1]] = image[min_y:max_y, min_x:max_x]
        # img = self.__bw_area_open(np.uint8(img), 5)
        # return img

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

    def __image_normalisation(self, image):
        try:
            y = 45
            if image.shape[0] == 0:
                return None
            kf = y / image.shape[0]
            x = int(image.shape[1] * kf)
            (ys, xs) = (64, 64)
            if x > xs:
                x = xs - 10
            image = cv2.resize(image, (x, y))
            imr = np.zeros((ys, xs))
            yo = int(0.5 * ys - image.shape[0] * 0.5)
            xo = int(0.5 * xs - image.shape[1] * 0.5)
            imr[yo:image.shape[0] + yo, xo:image.shape[1] + xo] = image[0:image.shape[0], 0:image.shape[1]]
            return imr
        except:
            logging.exception('')
            return None

    def __image_conversion(self, imglp, folder):
        try:
            imglp = np.squeeze(imglp, -1)
            result = self.__split_number(imglp, folder)
            return result
        except:
            logging.exception('')
            return None

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

    def __image_ocr_acc_level(self, images, acc_level=0.9):
        try:
            images = np.array(images) / 255
            images = np.reshape(images, images.shape + (1,))
            predict = self.ocrlp.predict(images)
            acc_class_ndx = np.where(predict >= acc_level)
            return acc_class_ndx[1]
        except:
            logging.exception('')
            return None

    def __image_pre_filter(self, image, fill_value=140, fill_value_zero=50, blur=3, blur_iter=3):
        try:
            img = image.copy()
            md = np.median(img)
            img[img >= md] = fill_value
            img[img < (md*0.6)] = fill_value_zero
            return cv2.GaussianBlur(img, (blur, blur), blur_iter)
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

    def recognize(self, image, folder):
        try:
            self.ok_ocr = False
            self.detect_number = []
            if not os.path.exists(folder):
                os.makedirs(folder)
            image_packet = self.__detect_lp(image)
            if image_packet is None:
                return
            for img in image_packet:
                if img is not None:
                    for im in img:
                        cv2.imwrite(os.path.join(folder, str(uuid.uuid4()) + '.jpg'), im)
                    img = self.__image_filter(img)
                    if img is not None:
                        for im in img:
                            cv2.imwrite(os.path.join(folder, str(uuid.uuid4()) + '.jpg'), im)
                        images_char = self.__image_split(img, folder)
                        self.__image_to_chars(images_char, folder)
            self.__select_ocr_number()
        except:
            logging.exception('')

    # split image to chars images
    def __image_split(self, images, folder):
        images_chars = []
        for image in images:
            image_chars = self.__image_conversion(image, folder)
            if image_chars is not None:
                images_chars.append(image_chars)
        return images_chars

    # ocr one iimage number
    def __image_to_chars(self, images, folder):
        for imgs in images:
            nclass = []
            if len(imgs) > 0:
                nclass = self.__image_ocr_acc_level(imgs, acc_level=self.cdl)
                i = 0
                for img in imgs:
                    if img is None:
                        continue
                    if i < len(nclass):
                        cv2.imwrite(os.path.join(folder,
                                                 self.letters[nclass[i]] + '_' + str(uuid.uuid4()) + '.jpg'), img)
                    i += 1
            number = ''
            if len(nclass) == 0:
                continue
            for i in range(len(nclass)):
                oc = self.letters[nclass[i]]
                if i > len(self.number_format):
                    continue
                number += self.__number_normalistion(oc, self.number_format[i] == 'd')
            f = open(os.path.join(folder, number + '.txt'), "a")
            f.write(number)
            f.close()
            print(number)
            if self.__match_to_number(number):
                self.detect_number.append(number)

    def __select_ocr_number(self):
        if len(self.detect_number) != 0:
            self.number_ocr = self.__max_number_detect(self.detect_number)[0]
            self.date_ocr = datetime.now()
            self.ok_ocr = True
            print('Detect number = %s' % self.number_ocr)

    def __max_number_detect(self, numbers):
        count = {}
        max_number = []
        for number in set(numbers):
            count[number] = numbers.count(number)
        for k, v in count.items():
            if v == max(count.values()):
                max_number.append(k)
        return max_number

    def image_filter(self, images):
        return self.__image_filter(images)

    def image_detect(self, image):
        try:
            img_crop = cv2.GaussianBlur(image.copy(), (5, 5), 1)
            img_crop = cv2.resize(img_crop, (224, 224))
            img_crop = img_crop / 255
            img_crop = np.reshape(img_crop, (1, img_crop.shape[0], img_crop.shape[1], 1))
            pred = self.dlp.predict(img_crop)[0]
            mask = np.zeros(pred.shape)
            mask[pred >= self.pdl] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = mask.astype(np.uint8)
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            result = []
            for cont in contours:
                epsilon = 0.005 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                area = cv2.contourArea(approx)
                print(x, y, h / w if w > 0 else 0, area)
                img_gepotise = []
                md_arr = []
                if w > 0 and h / w >= self.detect_k and area > self.detect_area:
                    for i in range(-10, 10):
                        out = mask[y - self.image_offset:y + h + self.image_offset,
                              x - self.image_offset:x + w + self.image_offset]
                        out = self.__image_rotate(out, i)
                        md = np.median(np.mean(out, axis=1))
                        md_arr.append(md)
                        out = image[y - self.image_offset:y + h + self.image_offset,
                              x - self.image_offset:x + w + self.image_offset]
                        out = self.__image_rotate(out, i)
                        out = cv2.resize(out, (256, 64))
                        img_gepotise.append(out)
                        result.append(out)
                    #if len(md_arr) == 0:
                    #    continue
                    mdmax = np.max(md_arr)
                    #maxs = np.where((md_arr >= np.uint32(mdmax)) & (md_arr <= mdmax))
                    #images = np.array(img_gepotise)[maxs]
                    img_packet = []
                    #for im in images:
                    #result.append(im)
                    #    img_packet.append(self.__image_pre_filter(im))
                     #   result.append(img_packet)
            return np.array(result)
        except:
            logging.exception('')
            return None

    def __number_normalistion(self, char, isdigist):
        if isdigist:
            return self.__char_to_dig(char)
        else:
            return self.__dig_to_char(char)

    def __dig_to_char(self, char):
        if char == '0':
            return 'O'
        if char == '8':
            return 'B'
        return char

    def __char_to_dig(self, char):
        if char == 'O':
            return '0'
        if char == 'B':
            return '8'
        return char

    def __match_to_number(self, number):
        regex = r"^\D\d{3}\D{2}\d{2,3}$"
        return re.findall(regex, number)

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
