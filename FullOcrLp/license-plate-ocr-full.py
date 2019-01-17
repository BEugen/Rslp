import os
import recognize
import cv2
import math
import videocapture
import numpy as np
import uuid
from multiprocessing import Process, Queue
import atexit
import time

IMG_PATH = '/mnt/misk/misk/lplate/images'
IMG_FOR_OCR = 'E:\\temp\\chars'
DETECT_LEVEL = 2

def ocr(qo, qi):
    rc = recognize.RecognizeLp()
    fn = os.path.join(IMG_FOR_OCR, str(uuid.uuid4()) + '.jpg')
    while True:
        if qi.qsize() > 0:
            image = qi.get()
            rc.recognize(image, fn)
            qo.put([rc.ok_ocr, rc.date_ocr, rc.number_ocr])
            print('End ocr')
        time.sleep(0.1)

def ocr_kill(ocr):
    ocr.terminate()

def main():
    img_old = None
    number = ''
    x1, x2, y1, y2 = (125, 950, 180, 700)#src='http:///mjpg/video.mjpg',
    (x_o, y_o) = 0, 0
    cap = cv2.VideoCapture('http://172.31.176.6/mjpg/video.mjpg')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    qo = Queue()
    qi = Queue()
    p = Process(target=ocr, args=(qo, qi))
    p.start()
    atexit.register(ocr_kill, p)
    while True:
        ret, image = cap.read()
        img = image[y1:y2, x1:x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (21, 21), 0)
        if img_old is None:
            img_old = blur
            continue
        #ld = np.sum((img.astype("float") - img_old.astype("float")) ** 2)
        #ld /= float(img.shape[0] * img.shape[1])
        #ld = round(ld / 100.0, 2)
        diff = cv2.absdiff(img_old, blur)
        img_old = blur
        ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imgc = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        (x, y, w, h) = 0, 0, 0, 0
        if len(contours) != 0:
            cnt = contours[0]
            max_area = cv2.contourArea(cnt)
            for cont in contours:
                if cv2.contourArea(cont) > max_area:
                    cnt = cont
                    max_area = cv2.contourArea(cont)
            epsilon = 0.025 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if h < 100 and w < 100:
                (x, y) = 0, 0
            else:
                cv2.rectangle(imgc, (x, y), (x + w, y + h), (0, 255, 0), 2)
        (xd, yd) = (math.fabs(x - x_o), math.fabs(y - y_o))
        cv2.rectangle(image, (0, 0), (image.shape[1], 50), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(round(xd, 2)) + ', ' + str(round(yd, 2)), (10, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        if (xd + yd + x + y) != 0.0:
            print(xd, yd, x, y)
        if 0 <= xd < 25 and 0 <= yd < 25 and (x != 0.0 or y != 0.0):
            print('Recognise!!!')
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            qi.put(img)
        (x_o, y_o) = x, y
        if qo.qsize() > 0:
            ocr_data = qo.get()
            if ocr_data[0]:
                number = ocr_data[2] + ' ' + ocr_data[1].strftime('%d.%m.%Y %H:%M.%S')
                cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 0, 255), 2)
        cv2.putText(image, number, (10, 90), font, 1, (255, 100, 0), 2, cv2.LINE_AA)
       # image[y1:y2, x1:x2] = imgc
        cv2.imshow('Video', image)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # for file in os.listdir(IMG_PATH):
    # #file = '16-12-2018_4-52-43_2.jpg'#'FirstFN_cam2 (176).jpg'
    #     fp = os.path.join(IMG_PATH, file)
    #     img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    #     x1, x2, y1, y2 = (125, 1070, 380, 830)
    #     img = img[y1:y2, x1:x2]
    #     fn = os.path.splitext(file)[0]
    #     print(fn)
    #     fn = os.path.join(IMG_FOR_OCR, fn)
    #     rc.recognize(img, fn)


if __name__ == '__main__':
    main()
