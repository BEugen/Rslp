import config
import os
import cv2
import numpy as np
import shutil
import random
import uuid

IMG_TO_AUG = '/mnt/misk/misk/lplate/augchar'
IMG_SRC = '/mnt/misk/misk/lplate/cchars'
IMG_COUT = 3000

def main():
    letters = config.LETTERS
    for l in letters:
        folder_dst = os.path.join(IMG_TO_AUG, l)
        if not os.path.exists(folder_dst):
            os.makedirs(folder_dst)
        folder_src = os.path.join(IMG_SRC, l)
        imc = len(os.listdir(folder_src))
        if imc >= IMG_COUT:
            files = os.listdir(folder_src)
            for i in range(0, IMG_COUT):
                shutil.copy(os.path.join(folder_src, files[i]), os.path.join(folder_dst, files[i]))
            continue
        else:
            files = os.listdir(folder_src)
            for i in range(0, imc):
                shutil.copy(os.path.join(folder_src, files[i]), os.path.join(folder_dst, files[i]))
            aug_count = IMG_COUT - imc
            for i in range(0, aug_count):
                file = random.choice(os.listdir(folder_src))
                img = cv2.imread(os.path.join(folder_src, file), cv2.IMREAD_GRAYSCALE)
                angle = random.uniform(-5.0, 0.5)
                img = image_rotate(img, angle)
                cv2.imwrite(os.path.join(folder_dst, str(uuid.uuid4()) + '.jpg'), img)



def image_rotate(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=0)
    return result

if __name__ == '__main__':
    main()