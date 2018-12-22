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
        self.letters = sorted(list(LP_LETTERS))
        self.max_len = LP_LETTERS
        self.pdl = PREDICT_DETECT_LEVEL
        self.dlp = self.__get_model_detect_lp()
        self.ocrlp = self.__get_model_ocr_lp()


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
            #print(md)
            #cv2.imwrite(str(i + 10) + '_' + str(round(md, 3)) + '_test.jpg', out)
            md_arr.append(md)
            out = img[min_y:max_y + 1, min_x:max_x + 1]
            out = self.__image_rotate(out, i)
            out = cv2.resize(out, (128, 64))
            # out = np.expand_dims(out.T, -1)/255
            img_gepotise.append(out)
        mdmax = np.max(md_arr)
        maxs = np.where((md_arr >= np.uint32(mdmax)) & (md_arr <= mdmax))
        return np.array(img_gepotise)[maxs]

    def __char_crop(self, img, mean_size=3, median_k_a=1.1, pix_shift_back=2, pix_shoft_forw=2,
                    char_size_min=6):
        mean_imgs = []
        imgs = []
        for i in range(0, img.shape[1]):
            mean_imgs.append(np.mean(img[:, i:i + mean_size]))
            i += mean_size
        med_all_img = np.median(mean_imgs) * median_k_a
        index = np.where(mean_imgs >= med_all_img)
        mean_imgs = np.array(mean_imgs)
        mean_imgs[index] = np.max(mean_imgs)
        index = np.where(mean_imgs >= np.max(mean_imgs))
        if len(index) > 0:
            index = index[0]
        if index[0] > (2 * mean_size):
            index = np.insert(index, 0, mean_size)
        for i in range(1, len(index)):
            if (index[i] - index[i - 1]) < char_size_min:
                continue
            li = index[i - 1] - pix_shift_back if index[i - 1] - pix_shift_back >= 0 else index[i - 1]
            ri = index[i] + pix_shoft_forw if index[i] + pix_shoft_forw <= img.shape[1] else img.shape[1]
            imgs.append(np.copy(img[0:img.shape[0], li:ri]))
        return imgs

    def __lp_crop(self, img, mean_size=2, max_crop=5):
        mean_imgs = []
        for i in range(0, img.shape[0]):
            mean_imgs.append(np.mean(img[i:i + mean_size, :]))
            i += mean_size
        med_all_img = np.mean(mean_imgs) * 0.8
        index = np.where(mean_imgs < med_all_img)
        hh = img.shape[0] * 0.5
        index = np.squeeze(index, -1)
        sl = index[index >= hh]
        if len(sl) > 0:
            l_top = np.min(sl)
            l_top = l_top if img.shape[0] - l_top <= max_crop else img.shape[0] - max_crop
            img[l_top: img.shape[0], :] = 255
        sl = index[index <= hh]
        if len(sl) > 0:
            l_bot = np.max(sl)
            l_bot = l_bot if l_bot <= max_crop else max_crop
            img[0:l_bot, :] = 255
        return img

    def __img_split(self, images, img_max_lenght=20):
        img_wb = []
        for i in range(0, len(images)):
            if images[i].shape[1] > img_max_lenght:
                img_wb.append(i)
        off_corr = 0
        for i in img_wb:
            imgs_split = self.__char_crop(images[i + off_corr], pix_shift_back=3, pix_shoft_forw=3)
            if len(imgs_split) > 0:
                images[i + off_corr] = imgs_split[0]
                inx = i + 1 + off_corr
                for y in range(1, len(imgs_split)):
                    images.insert(inx, imgs_split[y])
                    inx += 1
            off_corr += len(imgs_split) - 1
        return images

    #         cv2.imshow("Contour", cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))
    #         cv2.waitKey(2000)
    def __img_crop(self, img, mean_size=2, level_blank=2, invert=False):
        mean_imgs = []
        for i in range(0, img.shape[0]):
            mean_imgs.append(np.mean(img[i:i + mean_size, :]))
            i += mean_size
        index = np.where(mean_imgs >= np.int32(level_blank)) if invert else np.where(mean_imgs <= np.int32(level_blank))
        hh = img.shape[0] * 0.5
        index = np.squeeze(index, -1)
        sl = index[index >= hh]
        l_top = np.min(sl) - 1 if len(sl) > 0 else img.shape[0] - 1
        sl = index[index <= hh]
        l_bot = np.max(sl) + 1 if len(sl) > 0 else 1
        w = img.shape[1]
        h = l_top - l_bot
        imc = np.zeros((h, w))
        imc[:, :] = img[l_bot:l_top, :]
        return np.uint8(imc)

    def __img_crop_next(self, img, level_blank=5, axis=0):
        try:
            mean_imgs = np.mean(img, axis=axis)
            index = np.where(mean_imgs <= level_blank)
            if axis == 0:
                hw = int(np.mean(np.where(mean_imgs[20:44] == np.max(mean_imgs[20:44])))) + 20
            else:
                hw = img.shape[0] * 0.5
            shape_img = img.shape[1] if axis > 0 else img.shape[0]
            index = np.squeeze(index, -1)
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
        except:
            return None

    def __image_normalisation(self, image):
        try:
            koeff = 60 / image.shape[0]
            (h, w) = (int(image.shape[0]*koeff), int(image.shape[1]*koeff))
            image = 255-cv2.resize(image, (h, w))
            imr = np.zeros((64, 64))
            yo = int(0.5*64 - image.shape[0]*0.5)
            xo = int(0.5*64 - image.shape[1]*0.5)
            imr[yo:image.shape[0] + yo, xo:image.shape[1] + xo] = image[0:image.shape[0], 0:image.shape[1]]
            return imr
        except:
            return None

    def __image_conversion(self, image, lp_number):
        print(lp_number)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.adaptiveThreshold(image, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 1))
        ret, image = cv2.threshold(image, 100, 255, 0)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = self.__lp_crop(image)
        images = self.__char_crop(image)
        images = self.__img_split(images)
        self.__plot_images(images, True)
        for i in range(0, len(images)):
            img = self.__img_crop(images[i])
            if img is None:
                continue
            img = self.__img_crop(img, level_blank=251, invert=True)
            if img is None:
                continue
            img = self.__image_normalisation(img)
            if img is None:
                continue
            img = self.__img_crop_next(img)
            if img is None:
                continue
            img = self.__img_crop_next(img, axis=1)
            if img is None:
                continue
            imr = np.zeros((64, 64))
            yo = int(0.5 * 64 - img.shape[0] * 0.5)
            xo = int(0.5 * 64 - img.shape[1] * 0.5)
            imr[yo:img.shape[0] + yo, xo:img.shape[1] + xo] = img[0:img.shape[0], 0:img.shape[1]]
            images[i] = imr
        return images

    def __image_rotate(self, img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __image_ocr(self, images):
        images = np.array(images)/255
        images = np.reshape(images, images.shape + (1, ))
        predict = self.ocrlp.predict(images)
        return predict

    def recognize(self, image, file):
        img = self.__detect_lp(image, file)
        if img is not None:
            for im in img:
                images = self.__image_conversion(im, file)
                lp = self.__image_ocr(images)
            # self.__ocr_license_plate(img)
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

