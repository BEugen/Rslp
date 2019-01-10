import shutil
import os
import cv2
import numpy as np

ROOT_IMG = '/mnt/misk/misk/lplate/lp-un-mask'
IMG_PATH = '/mnt/misk/misk/lplate/lp-un-mask/img'
IMG_MASK = '/mnt/misk/misk/lplate/lp-un-mask/maskl'
SPLIT_SIZE = 0.25

def main():
    list_file = os.listdir(IMG_MASK)

    split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
    print(split_size)
    train = list_file[:split_size]
    test = list_file[split_size:]
    for file in train:
        img = mask_crop(IMG_MASK + '/' + file)
        cv2.imwrite(os.path.join(ROOT_IMG, 'train') + '/msk/msk_' + file, img)
        shutil.copy(IMG_PATH + '/' + file, os.path.join(ROOT_IMG, 'train') + '/img/img_' + file)

    for file in test:
        img = mask_crop(IMG_MASK + '/' + file)
        cv2.imwrite(os.path.join(ROOT_IMG, 'test') + '/msk/msk_' + file, img)
        shutil.copy(IMG_PATH + '/' + file, os.path.join(ROOT_IMG, 'test') + '/img/img_' + file)

def mask_crop(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgc = np.zeros((64, 286))
    imgc[:, :] = img[4:img.shape[0] - 4, :]
    return imgc

if __name__ == '__main__':
    main()

