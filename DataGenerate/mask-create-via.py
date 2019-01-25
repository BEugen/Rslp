import cv2, numpy as np
import skimage
from skimage import draw
import os
import json

IMG_STORE = '/mnt/misk/misk/lplate/data/data/train_1'
IMG_SOURCE = '/mnt/misk/misk/lplate/img-reg'
IMG_SIZE = (224, 224)


def main():
    js_path = os.path.join(IMG_SOURCE, 'via_region_data.json')
    json_file = open(js_path).read()
    js = json.loads(json_file)
    for imf in js:
        img_file = js[imf]['filename']
        img_p = os.path.join(IMG_SOURCE, img_file)
        if not os.path.exists(img_p):
            continue
        points = js[imf]['regions'][0]['shape_attributes']
        points_x = points['all_points_x']
        points_y = points['all_points_y']
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        img_dim = img.shape
        msk_img = mask_created(img_dim, (points_x, points_y))
        img = cv2.resize(img, IMG_SIZE)
        msk_img = cv2.resize(msk_img, IMG_SIZE)
        cv2.imwrite(os.path.join(IMG_STORE + '/msk', 'msk_' + img_file + '.jpg'), msk_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(IMG_STORE + '/img', 'img_' + img_file + '.jpg'), img,
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