import os
import recognize
import cv2

IMG_PATH = '/home/eugen/Загрузки/test-4850/archive/experiment001/AutoVis/train/img'
IMG_FOR_OCR = '/mnt/misk/misk/lplate/chars'


def main():
    rc = recognize.RecognizeLp()
    for file in os.listdir(IMG_PATH):
        fp = os.path.join(IMG_PATH, file)
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 64))
        fn = os.path.splitext(file)[0].split('_')
        rc.image_conversion(img, IMG_FOR_OCR, fn[len(fn)-1])


if __name__ == '__main__':
    main()
