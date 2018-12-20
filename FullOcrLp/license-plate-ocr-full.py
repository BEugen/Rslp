import os
import recognize
import cv2

IMG_PATH = 'E:/temp/images'
IMG_FOR_OCR = '/mnt/misk/misk/lplate/chars'


def main():
    rc = recognize.RecognizeLp()
    for file in os.listdir(IMG_PATH):
        fp = os.path.join(IMG_PATH, file)
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        fn = os.path.splitext(file)[0].split('_')
        rc.recognize(img, fn)


if __name__ == '__main__':
    main()
