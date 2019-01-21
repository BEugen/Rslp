import matplotlib.pyplot as plt
import recognize
import os
import cv2
import uuid

IMG_FOR_OCR = '/mnt/misk/misk/lplate/temp'

def main():
    rc = recognize.RecognizeLp()
    IMG_FOLDER = '/mnt/misk/misk/lplate/tstimg'
    for file in os.listdir(IMG_FOLDER):
        print(file)
        file = os.path.join(IMG_FOLDER, file)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        fn = os.path.join(IMG_FOR_OCR, str(uuid.uuid4()))
        if not os.path.exists(fn):
            os.makedirs(fn)
        rc.recognize(img, fn)


if __name__ == '__main__':
    main()