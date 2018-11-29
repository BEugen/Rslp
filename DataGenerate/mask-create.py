import cv2, numpy as np
import skimage
from skimage import draw
import os
import json


IMG_PATH = ''
IMG_MASK = ''

def main():
    for file in os.listdir(IMG_PATH + '/img'):
        pass


def dataset_img_create(img_path):
    pass


def mask_created(img_dim, points):
    img_mask = np.zeros(img_dim)
    rx, ry = points
    mr, mc = skimage.draw.polygon(ry, rx)
    img_mask[mr, mc] = 1
    return img_mask.astype(np.bool)


def get_points(gpath):
    pass


if __name__ == '__main__':
    main()

