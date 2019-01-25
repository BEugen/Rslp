import matplotlib.pyplot as plt
import recognize
import os
import cv2
import uuid
import numpy as np

IMG_FOR_OCR = 'E:/temp/chars'

def main():
    rc = recognize.RecognizeLp()
    IMG_FOLDER = 'E:/temp/test_img'
    for file in os.listdir(IMG_FOLDER):
        print(file)
        file = os.path.join(IMG_FOLDER, file)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        fn = os.path.join(IMG_FOR_OCR, str(uuid.uuid4()))
        if not os.path.exists(fn):
            os.makedirs(fn)
        rc.recognize(img, fn)

def test_filter():
    IMAGE_FILE = 'E:/temp/chars/6b719803-7468-40ea-87ef-3615a82b8a5f/0f573552-82ba-42fa-8877-e367e5c3c757.jpg'
    img = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
    rc = recognize.RecognizeLp()
    img = rc.image_filter([img])[0]
    img = np.squeeze(img, -1)
    plt.imshow(img, cmap='gray')
    plt.show()

def test_detect():
    IMAGE_FILE = 'E:/temp/chars/8ed99832-6c87-4ea6-9b3b-4891606d4150/2cd4bec0-54d1-464d-a286-72e72f83800d.jpg'
    img = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    rc = recognize.RecognizeLp()
    img = rc.image_detect(img)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    #img = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel)
    for im in img:
        plt.imshow(im, cmap='gray')
        plt.show()

if __name__ == '__main__':
    #main()
    test_detect()
