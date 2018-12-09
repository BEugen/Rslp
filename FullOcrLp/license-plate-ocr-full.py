import os
import recognize
import cv2

IMG_PATH = '/mnt/misk/misk/lplate/data/test_img_track'



def main():
    rc = recognize.RecognizeLp()
    for file in os.listdir(IMG_PATH):
        fp = os.path.join(IMG_PATH, file)
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        fn = os.path.splitext(file)[0]
        rc.recognize(img, fn)


if __name__ == '__main__':
    main()