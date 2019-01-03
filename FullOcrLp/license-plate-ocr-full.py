import os
import recognize
import cv2

IMG_PATH = '/mnt/misk/misk/lplate/images'
IMG_FOR_OCR = '/mnt/misk/misk/lplate/chars'


def main():
    rc = recognize.RecognizeLp()
    for file in os.listdir(IMG_PATH):
        fp = os.path.join(IMG_PATH, file)
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        x1, x2, y1, y2 = (125, 1070, 380, 830)
        img = img[y1:y2, x1:x2]
        fn = os.path.splitext(file)[0].split('_')
        rc.recognize(img, fn)


if __name__ == '__main__':
    main()
