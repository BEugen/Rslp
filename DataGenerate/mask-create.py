import cv2, numpy as np
import skimage
from skimage import draw
import os
import json


IMG_PATH = '/mnt/misk/misk/lplate/data'
IMG_MASK = '/mnt/misk/misk/lplate/data/msk'
IMG_SIZE = (224, 224)

def main():
    ant_path = os.path.join(IMG_PATH, 'ann')
    img_path = os.path.join(IMG_PATH, 'img')
    for file in os.listdir(ant_path):
        json_file = open(os.path.join(ant_path, file)).read()
        ann = json.loads(json_file)
        print(ann)
        points = get_points(ann['rectangle'])
        if points is None:
            continue
        img_dim = (ann['size']['height'], ann['size']['width'])
        msk_img = mask_created(img_dim, points)
        img = cv2.imread(os.path.join(img_path, os.path.splitext(file)[0] + '.bmp'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        msk_img = cv2.resize(msk_img, IMG_SIZE)
        cv2.imwrite(os.path.join(IMG_MASK, 'msk_' + os.path.splitext(file)[0] + '.jpg'), msk_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(IMG_PATH, 'jpg', 'img_' + os.path.splitext(file)[0] + '.jpg'), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])



def mask_created(img_dim, points):
    img_mask = np.zeros(img_dim)
    rx, ry = points
    mr, mc = skimage.draw.polygon(ry, rx)
    img_mask[mr, mc] = 255.0
    return img_mask


def get_points(gpath):
    rx = []
    ry = []
    for i in gpath:
        if len(i) == 0:
            return None
        rx.append(i[0])
        ry.append(i[1])
    return rx, ry


if __name__ == '__main__':
    main()

