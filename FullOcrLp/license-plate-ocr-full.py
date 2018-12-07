import os
import cv2
import recognize

IMG_PATH = 'E:/temp/images'


def main():
    rp = recognize.RecognizeLp()
    for file in os.listdir(IMG_PATH):
        file = os.path.join(IMG_PATH, file)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        rp.recognize(img)


if __name__ == '__main__':
    main()

