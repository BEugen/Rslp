import cv2
import os
import numpy as np


LP_LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
              'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
IMG_PATH_ROOT = '/mnt/misk/misk/lplate/chars'
IMG_TRAIN = '/mnt/misk/misk/lplate/chars/train'
IMG_TEST = '/mnt/misk/misk/lplate/chars/test'
SPLIT_SIZE = 0.25

#shape = 0 - h
#shape = 1 - w
#axis = 0 - columns
#axis = 1 - rows
def img_crop(img, level_blank = 5, axis=0):
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



def img_prepare(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    ret, img = cv2.threshold(img, 100, 255, 0)
    img = img_crop(img)
    if img is None:
        return None
    img = img_crop(img, axis=1)
    if img is None:
        return None
    imr = np.zeros((64, 64))
    yo = int(0.5 * 64 - img.shape[0] * 0.5)
    xo = int(0.5 * 64 - img.shape[1] * 0.5)
    imr[yo:img.shape[0] + yo, xo:img.shape[1] + xo] = img[0:img.shape[0], 0:img.shape[1]]
    return np.uint8(imr)


def img_write(path_read, path_write):
    img = img_prepare(path_read)
    if img is not None:
        cv2.imwrite(path_write, img)


def main():
    for icl in range(0, len(LP_LETTERS)):
        folder = os.path.join(IMG_PATH_ROOT, LP_LETTERS[icl])
        list_file = os.listdir(folder)

        split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
        print(split_size)
        train = list_file[:split_size]
        test = list_file[split_size:]
        for file in train:
            img_path = os.path.join(folder, file)
            file_result = os.path.splitext(file)[0].split('_')
            img_path_write = os.path.join(IMG_TRAIN, file_result[0] + '_' + str(icl) + '.jpg')
            img_write(img_path, img_path_write)

        for file in test:
            img_path = os.path.join(folder, file)
            file_result = os.path.splitext(file)[0].split('_')
            img_path_write = os.path.join(IMG_TEST, file_result[0] + '_' + str(icl) + '.jpg')
            img_write(img_path, img_path_write)


if __name__ == '__main__':
    main()
