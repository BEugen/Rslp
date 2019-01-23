import shutil
import os
import cv2
import numpy as np
import uuid

ROOT_IMG = '/mnt/misk/misk/lplate/lp-un-mask'
IMG_PATH = '/mnt/misk/misk/lplate/lp-un-mask/img'
IMG_MASK = '/mnt/misk/misk/lplate/lp-un-mask/maskl'
SPLIT_SIZE = 0.25


def __image_pre_filter(image, fill_value=140, blur=3, blur_iter=3):
    img = image.copy()
    md = np.median(img)
    img[img >= md] = fill_value
    return cv2.GaussianBlur(img, (blur, blur), blur_iter)


def main():
    dir = os.listdir(IMG_MASK)
    for file in dir:
        im = cv2.imread(os.path.join(IMG_PATH, file), cv2.IMREAD_GRAYSCALE)
        im = __image_pre_filter(im)
        new_filename = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(os.path.join(IMG_PATH, new_filename), im)
        shutil.copy(os.path.join(IMG_MASK, file), os.path.join(IMG_MASK, new_filename))


if __name__ == '__main__':
    main()