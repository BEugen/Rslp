import os
import recognize
import cv2
import videocapture
import numpy as np
import uuid
from multiprocessing import Process, Queue
import atexit

IMG_PATH = '/mnt/misk/misk/lplate/images'
IMG_FOR_OCR = '/mnt/misk/misk/lplate/test_img_all'
DETECT_LEVEL = 50

def ocr(qo, qi):
    rc = recognize.RecognizeLp()
    fn = os.path.join(IMG_FOR_OCR, str(uuid.uuid4()) + '.jpg')
    while True:
        if qi.qsize() > 0:
            image = qi.get()
            rc.recognize(image, fn)
            qo.put([rc.ok_ocr, rc.date_ocr, rc.number_ocr])
            print('End ocr')

def ocr_kill(ocr):
    ocr.terminate()

def main():
    img_old = None

    x1, x2, y1, y2 = (125, 950, 280, 700)#src='http:///mjpg/video.mjpg',
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    qo = Queue()
    qi = Queue()
    p = Process(target=ocr, args=(qo, qi))
    p.start()
    atexit.register(ocr_kill, p)
    while True:
        ret, image = cap.read()
        img = image[y1:y2, x1:x2]
        if img_old is None:
            img_old = img
            continue
        ld = np.sum((img.astype("float") - img_old.astype("float")) ** 2)
        ld /= float(img.shape[0] * img.shape[1])
        ld = round(ld / 100.0, 2)
        img_old = img
        cv2.rectangle(image, (0, 0), (image.shape[1], 50), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(ld), (10, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        if ld > DETECT_LEVEL:
            print('Recognise!!!')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            qi.put(img)
        if qo.qsize() > 0:
            ocr_data = qo.get()
            if ocr_data[0]:
                cv2.putText(image, ocr_data[2], (50, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, ocr_data[1].strftime('%d.%m.%Y %H:%M.%S'), (100, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 0, 255), 2)
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
